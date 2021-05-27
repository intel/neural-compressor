#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import copy
import logging
from collections import OrderedDict
import yaml
import numpy as np
from lpot.adaptor.adaptor import adaptor_registry, Adaptor
from lpot.adaptor.query import QueryBackendCapability
from lpot.utils.utility import LazyImport, dump_elapsed_time

onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")

logger = logging.getLogger()

class ONNXRTAdaptor(Adaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.logger = logger
        self.static = framework_specific_info["approach"] == "post_training_static_quant"
        self.backend = framework_specific_info["backend"]
        self.work_space = framework_specific_info["workspace_path"]
        os.makedirs(self.work_space, exist_ok=True)
        self.pre_optimized_model = None
        self.quantizable_op_types = self._query_quantizable_op_types()
        self.evaluate_nums = 0

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.q_config = {} # adaptor should know current configs at any time
        self.quantize_params = {} # adaptor should know current params at any time

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            data_loader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for onnx.

        Returns:
            (dict): quantized model
        """
        assert q_func is None, "quantization aware training has not been supported on ONNXRUNTIME"
        model = self.pre_optimized_model if self.pre_optimized_model else model
        ort_version = [int(i) for i in ort.__version__.split(".")]
        if ort_version < [1, 5, 2]:
            logger.warning('quantize input need onnxruntime version > 1.5.2')
            return model
        if model.model.opset_import[0].version < 11:
            logger.warning('quantize input need model opset >= 11')
        from lpot.adaptor.ox_utils.onnx_quantizer import ONNXQuantizer
        from onnxruntime.quantization.quant_utils import QuantizationMode
        backend = QuantizationMode.QLinearOps if self.backend == \
            "qlinearops" else QuantizationMode.IntegerOps
        model = copy.deepcopy(model)
        self.quantizable_ops = self._query_quantizable_ops(model.model)
        q_config = self._cfg_to_qconfig(tune_cfg)
        iterations = tune_cfg.get('calib_iteration', 1)
        if self.static:
            quantize_params = self._get_quantize_params(model.model, data_loader, q_config, \
                                                                                  iterations)
        else:
            quantize_params = None
        self.quantize_params = quantize_params
        quantizer = ONNXQuantizer(model.model,
            q_config,
            backend,
            self.static,
            quantize_params,
            self.quantizable_op_types)
        quantizer.quantize_model()
        model.model = quantizer.model.model
        self.q_config = q_config # update so other methods can know current configs
        return model

    def _get_quantize_params(self, model, data_loader, q_config, iterations):
        from lpot.adaptor.ox_utils.onnxrt_mid import ONNXRTAugment
        from lpot.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        black_nodes = [node for node in q_config if q_config[node]=='fp32']
        white_nodes = [node for node in q_config if q_config[node]!='fp32']
        augment = ONNXRTAugment(model, \
                  data_loader, self.quantizable_op_types, \
                  os.path.join(self.work_space, 'augmented_model.onnx'), \
                  black_nodes=black_nodes, white_nodes=white_nodes, \
                  iterations=list(range(0, q_config['calib_iteration'])))
        quantize_params = augment.dump_calibration()
        return quantize_params

    def inspect_tensor(self, model, data_loader, op_list=[],
                       iteration_list=[],
                       inspect_type='activation',
                       save_to_disk=False):
        '''The function is used by tune strategy class for dumping tensor info.
        '''
        from lpot.adaptor.ox_utils.onnxrt_mid import ONNXRTAugment
        from lpot.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        augment = ONNXRTAugment(model, data_loader, [], \
                  os.path.join(self.work_space, 'augment_for_inspect.onnx'), \
                  iterations=iteration_list,
                  white_nodes=op_list)
        tensors = augment.dump_tensor(activation=(inspect_type!='weight'),
                                      weight=(inspect_type!='activation'))
        if save_to_disk:
            np.savez(tensors, os.path.join(self.work_space, 'dumped_tensors.npz'))
        return tensors

    def set_tensor(self, model, tensor_dict):
        from onnx import numpy_helper
        from lpot.model.onnx_model import ONNXModel
        from lpot.adaptor.ox_utils.util import quantize_data_with_scale_zo
        from lpot.adaptor.ox_utils.util import quantize_data_per_channel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        assert "QuantizeLinear" in [node.op_type for node in model.model.graph.node], \
                                           'adaptor.set_tensor only accept int8 model'
        input_name_to_nodes = model.input_name_to_nodes
        for tensor_name, tensor_value in tensor_dict.items():
            if not tensor_name.endswith('_quantized'):
                tensor_name += '_quantized'
            not_filter = False
            scale_tensor, zo_tensor = model.get_scale_zo(tensor_name)
            if scale_tensor is None or zo_tensor is None:
                not_filter = True
            else:
                scale_value = numpy_helper.to_array(scale_tensor)
                zo_value = numpy_helper.to_array(zo_tensor)
            assert len(input_name_to_nodes[tensor_name]) == 1, \
                    'quantized filter weight should be input of only one node'
            node = input_name_to_nodes[tensor_name][0] #TBD only for conv bias
            node_name = node.name.replace('_quant', '')
            assert node_name in self.q_config
            q_type = self.q_config[node_name]['weight']['dtype']
            if not_filter:
                new_tensor_value = self._requantize_bias(model, tensor_name, tensor_value)
            elif self.q_config[node_name]['weight']['granularity'] == 'per_tensor':
                new_tensor_value = quantize_data_with_scale_zo(tensor_value,
                                                               q_type,
                                                               scale_value,
                                                               zo_value)
            else:
                new_tensor_value = quantize_data_per_channel(tensor_value,
                                                             q_type,
                                                             scale_value,
                                                             zo_value)
            model.set_initializer(tensor_name, new_tensor_value)
        return model

    def _requantize_bias(self, model, bias_name, bias_data):
        ''' helper function to requantize bias, borrowed from onnx_quantizer '''
        from onnx import numpy_helper
        node = model.input_name_to_nodes[bias_name][0]
        input_scale_name = node.input[1]
        input_scale = numpy_helper.to_array(model.get_initializer(input_scale_name))

        weight_scale_name = node.input[4]
        weight_scale = numpy_helper.to_array(model.get_initializer(weight_scale_name))

        bias_scale = input_scale * weight_scale
        new_bias_data = (bias_data / bias_scale).round().astype(np.int32)
        return new_bias_data

    def _pre_optimize(self, model, level=1):
        sess_options = ort.SessionOptions()
        level = self.query_handler.get_graph_optimization() # pylint: disable=no-member
        sess_options.graph_optimization_level = level
        sess_options.optimized_model_filepath = os.path.join(self.work_space, \
            "Optimized_model.onnx")
        _ = ort.InferenceSession(model.model.SerializeToString(), sess_options)
        tmp_model = onnx.load(sess_options.optimized_model_filepath)
        model.model = self._replace_gemm_with_matmul(tmp_model).model
        self.pre_optimized_model = model

    def _replace_gemm_with_matmul(self, model):
        new_nodes = []
        from onnx import numpy_helper
        from lpot.model.onnx_model import ONNXModel
        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)

        for node in model.nodes():
            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B = model.get_initializer(node.input[1])
                        if B:
                            # assume B is not used by any other node
                            B_array = numpy_helper.to_array(B)
                            B_trans = numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            model.remove_initializer(B)
                            model.add_initializer(B_trans)

                            #TBD this is for onnx model zoo, which are all in old IR version
                            if model.model.ir_version < 4:
                                for input in model.model.graph.input:
                                    if input.name == B_trans.name:
                                        for i, dim in enumerate(input.type.tensor_type.shape.dim):
                                            dim.dim_value = B_array.T.shape[i]

                        else:
                            inputB += '_Transposed'
                            transpose_node = onnx.helper.make_node('Transpose',
                                                                inputs=[node.input[1]],
                                                                outputs=[inputB],
                                                                name=node.name+'_Transpose')
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node('MatMul',
                            inputs=[node.input[0], inputB],
                            outputs=[node.output[0] + ('_MatMul' if len(node.input)>2 else '')],
                            name=node.name + '_MatMul')
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node('Add',
                            inputs=[node.output[0] + '_MatMul', node.input[2]],
                            outputs=node.output,
                            name=node.name + '_Add')
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        model.graph().ClearField('node')
        model.graph().node.extend(new_nodes)

        return model

    def query_fw_capability(self, model):
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: onnx model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        self._pre_optimize(model)
        quantizable_ops = self._query_quantizable_ops(self.pre_optimized_model.model)
        optype_wise = OrderedDict()
        special_config_types = list(self.query_handler.get_quantization_capability()\
                                     ['int8'].keys())  # pylint: disable=no-member
        default_config = self.query_handler.get_quantization_capability()[\
                                     'int8']['default'] # pylint: disable=no-member
        op_wise = OrderedDict()
        for _, op in enumerate(quantizable_ops):
            if op.op_type not in special_config_types:
                op_capability = default_config
            else:
                op_capability = \
                    self.query_handler.get_quantization_capability()[\
                                   'int8'][op.op_type]  # pylint: disable=no-member
            if op.op_type not in optype_wise.keys():
                optype_wise[op.op_type] = copy.deepcopy(op_capability)

            op_wise.update(
                {(op.name, op.op_type): copy.deepcopy(op_capability)})

        return {'optypewise': optype_wise, 'opwise': op_wise}

    def _cfg_to_qconfig(self, tune_cfg):
        q_config = {}
        q_config['calib_iteration'] = tune_cfg['calib_iteration']
        granularity = 'per_tensor'
        algorithm = 'minmax'
        scheme = 'sym'

        from onnx import onnx_pb as onnx_proto
        for _, op in enumerate(self.quantizable_ops):
            if tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['dtype'] == 'fp32':
                q_config[op.name] = 'fp32'
            else:
                node_config = copy.deepcopy(tune_cfg['op'][(op.name, op.op_type)])
                for tensor, config in tune_cfg['op'][(op.name, op.op_type)].items():
                    if 'granularity' not in config:
                        node_config[tensor]['granularity'] = granularity
                    if 'algorithm' not in config:
                        node_config[tensor]['algorithm'] = algorithm
                    if 'scheme' not in config:
                        node_config[tensor]['scheme'] = scheme
                    if config['dtype'] == "int8":
                        node_config[tensor]['dtype'] = \
                                  onnx_proto.TensorProto.INT8 # pylint: disable=no-member
                    else:
                        node_config[tensor]['dtype'] = \
                                 onnx_proto.TensorProto.UINT8 # pylint: disable=no-member
                q_config[op.name] = node_config

        return q_config

    def _query_quantizable_ops(self, model):
        for node in model.graph.node:
            if node.op_type in self.quantizable_op_types:
                self.quantizable_ops.append(node)

        return self.quantizable_ops

    def _query_quantizable_op_types(self):
        quantizable_op_types = self.query_handler.get_op_types_by_precision( \
                                  precision='int8') # pylint: disable=no-member

        return quantizable_op_types

    def evaluate(self, input_graph, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """The function is for evaluation if no given eval func

        Args:
            input_graph      : onnx model for evaluation
            dataloader       : dataloader for evaluation. lpot.data.dataloader.ONNXDataLoader
            postprocess      : post-process for evalution. lpot.data.transform.ONNXTransforms
            metrics:         : metrics for evaluation. lpot.metric.ONNXMetrics
            measurer         : lpot.objective.Measurer
            iteration(int)   : max iterations of evaluaton.
            tensorboard(bool): whether to use tensorboard for visualizaton
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            (float) evaluation results. acc, f1 e.g.
        """
        sess_options = ort.SessionOptions()
        if measurer:
            # https://github.com/microsoft/onnxruntime/issues/7347
            cores_per_instance = int(os.environ.get('CORES_PER_INSTANCE'))
            assert cores_per_instance > 0, "benchmark cores_per_instance should greater than 0"
            sess_options.intra_op_num_threads = cores_per_instance
        session = ort.InferenceSession(input_graph.model.SerializeToString(), sess_options)
        if metric:
            metric.reset()
            if hasattr(metric, "compare_label") and not metric.compare_label:
                self.fp32_preds_as_label = True
                results = []

        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        for idx, batch in enumerate(dataloader):
            labels = batch[1]
            if measurer is not None:
                for i in range(len_inputs):
                    # in case dataloader contains non-array input
                    if not isinstance(batch[i], np.ndarray):
                        ort_inputs.update({inputs_names[i]: np.array(batch[i])})
                    else:
                        ort_inputs.update({inputs_names[i]: batch[i]})
                measurer.start()
                predictions = session.run(None, ort_inputs)
                measurer.end()
            else:
                for i in range(len_inputs):
                    ort_inputs.update({inputs_names[i]: batch[i]})
                predictions = session.run(None, ort_inputs)

            if self.fp32_preds_as_label:
                self.fp32_results.append(predictions) if fp32_baseline else \
                    results.append(predictions)

            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metric is not None and not self.fp32_preds_as_label:
                metric.update(predictions, labels)
            if idx + 1 == iteration:
                break

        if self.fp32_preds_as_label:
            from lpot.adaptor.ox_utils.util import collate_preds
            if fp32_baseline:
                results = collate_preds(self.fp32_results)
                metric.update(results, results)
            else:
                reference = collate_preds(self.fp32_results)
                results = collate_preds(results)
                metric.update(results, reference)

        acc = metric.result() if metric is not None else 0
        return acc

    def save(self, model, path):
        """ save model

        Args:
            model (ModelProto): model to save
            path (str): save path
        """
        model.save(os.path.join(path, "best_model.onnx"))


@adaptor_registry
class ONNXRT_QLinearOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_qlinear.yaml"))
        self.backend = "qlinearops"
        super().__init__(framework_specific_info)


@adaptor_registry
class ONNXRT_IntegerOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        self.query_handler = ONNXRTQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "onnxrt_integer.yaml"))
        self.backend = "integerops"
        super().__init__(framework_specific_info)

class ONNXRTQuery(QueryBackendCapability):

    def __init__(self, local_config_file=None):
        super().__init__()
        self.version = ort.__version__
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e: # pragma: no cover
                self.logger.info("Failed to parse {} due to {}".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check the {} format.".format(self.cfg))

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        default_config = None
        for sub_data in data:
            if sub_data['version']['name'] == self.version:
                return sub_data

            if sub_data['version']['name'] == 'default':
                default_config = sub_data

        return default_config

    def get_version(self):
        """Get the current backend version infomation.

        Returns:
            [string]: version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self):
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return self.cur_config['precisions']['names']

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config['ops']

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

    def get_op_types_by_precision(self, precision):
        """Get op types per precision

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in list(self.cur_config['ops'].keys())

        return self.cur_config['ops'][precision]

    def get_graph_optimization(self):
        """ Get onnxruntime graph optimization level"""
        optimization_levels = {'DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                               'ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                               'ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                               'ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
        level = self.cur_config['graph_optimization']['level']
        assert level in optimization_levels, "the optimization choices \
                                              are {}".format(optimization_levels.keys())
        return optimization_levels[level]

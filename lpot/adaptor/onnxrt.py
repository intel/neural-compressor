#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

import numpy as np
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport

onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")

logger = logging.getLogger()

class ONNXRTAdaptor(Adaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super(ONNXRTAdaptor, self).__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.logger = logger
        self.static = framework_specific_info["approach"] == "post_training_static_quant"
        self.backend = framework_specific_info["backend"]
        self.work_space = framework_specific_info["workspace_path"]
        self.pre_optimized_model = None
        self.quantizable_op_types = self._query_quantizable_op_types()

    def quantize(self, tune_cfg, model, dataLoader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for onnx.

        Returns:
            (dict): quantized model
        """
        model = self.pre_optimized_model if self.pre_optimized_model else model
        ort_version = [int(i) for i in ort.__version__.split(".")]
        if ort_version < [1, 5, 2]:
            logger.warning('quantize input need onnxruntime version > 1.5.2')
            return model
        if model.opset_import[0].version < 11:
            logger.warning('quantize input need model opset >= 11')
        
        from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
        from onnxruntime.quantization.quant_utils import QuantizationMode
        backend = QuantizationMode.QLinearOps if self.backend == \
            "qlinearops" else QuantizationMode.IntegerOps
        model = copy.deepcopy(model)
        iterations = tune_cfg.get('calib_iteration', 1)
        self.quantizable_ops = self._query_quantizable_ops(model)
        q_config = self._cfg_to_qconfig(tune_cfg)
        if self.static:
            quantize_params = self._get_quantize_params(model, dataLoader, q_config, iterations)
        else:
            quantize_params = None
        quantizer = ONNXQuantizer(model,
            q_config["per_channel"],
            q_config["reduce_range"],
            backend,
            self.static,
            q_config["weight_dtype"],
            q_config["input_dtype"],
            quantize_params,
            q_config["nodes_include"],
            q_config["nodes_exclude"],
            self.quantizable_op_types)
        quantizer.quantize_model()
        return quantizer.model.model

    def _get_quantize_params(self, model, dataloader, q_config, iterations):
        from .ox_utils.onnx_calibrate import calibrate
        quantize_params = calibrate(model, dataloader, self.quantizable_op_types, \
            q_config["nodes_exclude"], q_config["nodes_include"], iterations=iterations)
        return quantize_params

    def _pre_optimize(self, model, level=1):
        # TODO hardcoded to GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = os.path.join(self.work_space, \
            "Optimized_model.onnx")
        session = ort.InferenceSession(model.SerializeToString(), sess_options)
        self.pre_optimized_model = onnx.load(sess_options.optimized_model_filepath)

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
        quantizable_ops = self._query_quantizable_ops(self.pre_optimized_model)
        optype_wise = OrderedDict()
        op_wise = OrderedDict()
        for _, op in enumerate(quantizable_ops):
            optype = op.op_type
            op_capability = {
                'activation': {
                    'dtype': ['uint8', 'fp32']},
                'weight': {
                    'dtype': ['int8', 'fp32']}
            }
            if optype not in optype_wise.keys():
                optype_wise[optype] = op_capability

            op_wise.update(
                {(op.name, optype): op_capability})

        return {'optypewise': optype_wise, 'opwise': op_wise}

    def _cfg_to_qconfig(self, tune_cfg):
        nodes_exclude = []
        nodes_include = []
        weight_dtype = None
        input_dtype = None
        per_channel = None

        for _, op in enumerate(self.quantizable_ops):
            if tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['dtype'] == 'fp32':
                nodes_exclude.append(op.name)
            else:
                nodes_include.append(op.name)
                if weight_dtype:
                    assert weight_dtype == tune_cfg['op'][(op.name, op.op_type)
                              ]['weight']['dtype']
                weight_dtype = tune_cfg['op'][(op.name, op.op_type)
                              ]['weight']['dtype']
                if input_dtype:
                    assert input_dtype == tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['dtype']
                input_dtype = tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['dtype']
                if per_channel:
                    assert per_channel == tune_cfg['op'][(op.name, op.op_type)
                              ]['activation']['granularity']
                per_channel = tune_cfg['op'][(op.name, op.op_type)
                              ]['weight']['granularity']

        from onnx import onnx_pb as onnx_proto
        q_config = {}
        q_config["per_channel"] = per_channel
        q_config["reduce_range"] = False
        q_config["weight_dtype"] = onnx_proto.TensorProto.INT8 if weight_dtype == "int8" \
            else onnx_proto.TensorProto.UINT8
        q_config["input_dtype"] = onnx_proto.TensorProto.INT8 if input_dtype == "int8" \
            else onnx_proto.TensorProto.UINT8
        q_config["nodes_include"] = nodes_include
        q_config["nodes_exclude"] = nodes_exclude

        return q_config

    def _query_quantizable_ops(self, model):
        for node in model.graph.node:
            if node.op_type in self.quantizable_op_types:
                self.quantizable_ops.append(node)

        return self.quantizable_ops

    def _query_quantizable_op_types(self):
        # TBD, we exclude "gather" for static quantize
        # will be replaced with FWK query api
        if self.backend == "qlinearops":
            quantizable_op_types = ['Conv', 'MatMul', 'Attention', 'Mul', 'Relu', 'Clip', \
                'LeakyRelu', 'Gather', 'Sigmoid', 'MaxPool', 'EmbedLayerNormalization']
        else:
            quantizable_op_types = ['Gather', 'MatMul', 'Attention', \
                'EmbedLayerNormalization']
        return quantizable_op_types

    def evaluate(self, input_graph, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1, tensorboard=False):
        """The function is for evaluation if no given eval func

        Args:
            input_graph      : onnx model for evaluation
            dataloader       : dataloader for evaluation. lpot.data.dataloader.ONNXDataLoader
            postprocess      : post-process for evalution. lpot.data.transform.ONNXTransforms
            metrics:         : metrics for evaluation. lpot.metric.ONNXMetrics
            measurer         : lpot.objective.Measurer
            iteration(int)   : max iterations of evaluaton.
            tensorboard(bool): whether to use tensorboard for visualizaton

        Returns:
            (float) evaluation results. acc, f1 e.g.
        """
        session = ort.InferenceSession(input_graph.SerializeToString(), None)

        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        for idx, batch in enumerate(dataloader):
            labels = batch[-1]
            if measurer is not None:
                for i in range(len_inputs):
                    # in case dataloader contains non-array input
                    if not isinstance(batch[i], np.ndarray):
                        ort_inputs.update({inputs_names[i]: np.array(batch[i])})
                    else:
                        ort_inputs.update({inputs_names[i]: batch[i]})
                measurer.start()
                predictions = session.run([], ort_inputs)
                measurer.end()
            else:
                for i in range(len_inputs):
                    ort_inputs.update({inputs_names[i]: batch[i]})
                predictions = session.run([], ort_inputs)
            predictions = predictions[0] if len(predictions) == 1 else predictions

            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metric is not None:
                metric.update(predictions, labels)
            if idx + 1 == iteration:
                break
        acc = metric.result() if metric is not None else 0
        return acc

    def save(self, model, path):
        onnx.save_model(model, os.path.join(path, "best_model.onnx"))


@adaptor_registry
class ONNXRT_QLinearOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super(ONNXRT_QLinearOpsAdaptor, self).__init__(framework_specific_info)


@adaptor_registry
class ONNXRT_IntegerOpsAdaptor(ONNXRTAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super(ONNXRT_IntegerOpsAdaptor, self).__init__(framework_specific_info)


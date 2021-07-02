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
from collections import OrderedDict
import yaml
import numpy as np
from .query import QueryBackendCapability
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, CpuInfo, singleton, Dequantize, dump_elapsed_time
from ..utils import logger
from ..conf.dotdict import deep_get
tensorflow = LazyImport('tensorflow')


@adaptor_registry
class TensorFlowAdaptor(Adaptor):
    unify_op_type_mapping = {
        "Conv2D": "conv2d",
        "DepthwiseConv2dNative": "conv2d",
        "MaxPool": "pooling",
        "AvgPool": "pooling",
        "ConcatV2": "concat",
        "MatMul": "matmul",
        "Pad": "pad"
    }

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)

        self.quantize_config = {'op_wise_config': {}}
        self.framework_specific_info = framework_specific_info
        self.device = self.framework_specific_info['device']
        self.work_dir = os.path.abspath(self.framework_specific_info['workspace_path'])
        self.recipes = deep_get(self.framework_specific_info, 'recipes', {})
        self.optimization = deep_get(self.framework_specific_info, 'optimization', {})
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.pre_optimized_model = None
        self.pre_optimizer_handle = None

        self.bf16_ops = []
        self.fp32_ops = []
        self.dump_times = 0   # for tensorboard
        self.query_handler = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "tensorflow.yaml"))
        self.op_wise_sequences = self.query_handler.get_eightbit_patterns()
        self.optimization = self.query_handler.get_grappler_optimization_cfg()

        self.fp32_results = []
        self.fp32_preds_as_label = False

    def log_histogram(self, writer, tag, values, step=0, bins=1000):
        import tensorflow as tf
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)
        writer.flush()

    def evaluate(self, model, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        """Evaluate the model for specified metric on validation dataset.

        Args:
            model ([Graph, GraphDef or Path String]): The model could be the graph,
                        graph_def object, the frozen pb or ckpt/savedmodel folder path.
            dataloader (generator): generate the data and labels.
            postprocess (object, optional): process the result from the model
            metric (object, optional): Depends on model category. Defaults to None.
            measurer (object, optional): for precise benchmark measurement.
            iteration(int, optional): control steps of mini-batch
            tensorboard (boolean, optional): for tensorboard inspect tensor.
            fp32_baseline (boolen, optional): only for compare_label=False pipeline

        Returns:
            [float]: evaluation result, the larger is better.
        """
        import tensorflow as tf
        from .tf_utils.util import iterator_sess_run
        outputs = model.output_tensor_names

        if tensorboard:
            from .tf_utils.graph_rewriter.graph_util import GraphAnalyzer
            from tensorflow.python.framework import tensor_util

            output_postfix = "_fp32.output"
            inspect_node_types = ["Conv2D", "DepthwiseConv2dNative", "MaxPool", "AvgPool",
                                  "ConcatV2", "MatMul", "FusedBatchNormV3", "BiasAdd",
                                  "Relu", "Relu6", "Dequantize"]
            fp32_inspect_node_name = []
            int8_inspect_node_name = []
            q_node_scale = {}
            if self.dump_times == 0:
                temp_dir = "./runs/eval/baseline"
            else:
                temp_dir = "./runs/eval/tune_" + str(self.dump_times)
            if os.path.isdir(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            writer = tf.compat.v1.summary.FileWriter(temp_dir, model.graph)

            cur_graph = GraphAnalyzer()
            cur_graph.graph = model.graph_def
            cur_graph.parse_graph()
            graph_info = cur_graph.node_name_details
            for node in model.graph_def.node:
                if node.op in inspect_node_types:
                    fp32_inspect_node_name.append(node.name)
                # Tensor dump supported quantized op including,
                # Requantize, QuantizedConv2DAndRequantize,
                # QuantizedConv2DAndReluAndRequantize,
                # QuantizedConv2DWithBiasAndRequantize,
                # QuantizedConv2DWithBiasAndReluAndRequantize,
                # QuantizedConv2DWithBiasSignedSumAndReluAndRequantize,
                # QuantizedConv2DWithBiasSumAndReluAndRequantize,
                # QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize,
                # QuantizedMatMulWithBiasAndReluAndRequantize,
                # QuantizedMatMulWithBiasAndRequantize
                elif node.op.find("Requantize") != -1:
                    out_min = -2
                    out_max = -1
                    if node.op.find("Sum") != -1:
                        out_min = -5
                        out_max = -4
                    q_out_min = graph_info[node.input[out_min]
                                           ].node.attr["value"].tensor.float_val[0]
                    q_out_max = graph_info[node.input[out_max]
                                           ].node.attr["value"].tensor.float_val[0]
                    q_node_scale[node.name] = (node.op, q_out_min, q_out_max)
                    int8_inspect_node_name.append(node.name)
                # Inspect weights, bias. Need further optimize
                if node.op == "Const" and graph_info[graph_info[node.name].outputs[0]].node.op \
                    in ["Conv2D", "DepthwiseConv2dNative", "MatMul",
                    "FusedBatchNormV3", "BiasAdd"]:
                    const_value = tensor_util.MakeNdarray(node.attr.get(
                                  'value').tensor).astype(np.float32)
                    self.log_histogram(writer, node.name, const_value)

            outputs.extend(fp32_inspect_node_name)
            if len(int8_inspect_node_name) > 0:
                output_postfix = "_int8.output"
                outputs.extend(int8_inspect_node_name)

        if metric:
            metric.reset()
            if hasattr(metric, "compare_label") and not metric.compare_label:
                self.fp32_preds_as_label = True
                results = []

        origin_output_tensor_names = model.output_tensor_names
        model.output_tensor_names = outputs
        input_tensor = model.input_tensor
        output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                            model.output_tensor[0]
        logger.info("Start to evaluate Tensorflow model...")
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            if len(input_tensor) == 1:
                feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), \
                    'inputs len must equal with input_tensor'
                feed_dict = dict(zip(input_tensor, inputs))

            if model.iter_op:
                predictions = iterator_sess_run(model.sess, model.iter_op, \
                    feed_dict, output_tensor, iteration, measurer)
            elif measurer is not None:
                measurer.start()
                predictions = model.sess.run(output_tensor, feed_dict)
                measurer.end()
            else:
                predictions = model.sess.run(output_tensor, feed_dict)

            if self.fp32_preds_as_label:
                self.fp32_results.append(predictions) if fp32_baseline else \
                    results.append(predictions)

            # Inspect node output, just get 1st iteration output tensors for now
            if idx == 0 and tensorboard:
                for index, node_name in enumerate(outputs):
                    tensor = predictions[index]
                    if node_name in int8_inspect_node_name:
                        tensor = Dequantize(predictions[index], q_node_scale[node_name])
                    self.log_histogram(writer, node_name + output_postfix, tensor.astype(
                                       np.float32), idx)
                writer.close()
            if isinstance(predictions, list):
                if len(origin_output_tensor_names) == 1:
                    predictions = predictions[0]
                elif len(origin_output_tensor_names) > 1:
                    predictions = predictions[:len(origin_output_tensor_names)]
            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metric is not None and not self.fp32_preds_as_label:
                metric.update(predictions, labels)
            if idx + 1 == iteration:
                break

        if self.fp32_preds_as_label:
            from .tf_utils.util import collate_tf_preds
            if fp32_baseline:
                results = collate_tf_preds(self.fp32_results)
                metric.update(results, results)
            else:
                reference = collate_tf_preds(self.fp32_results)
                results = collate_tf_preds(results)
                metric.update(results, reference)

        acc = metric.result() if metric is not None else 0
        if tensorboard:
            new_dir = temp_dir + "_acc_" + str(acc)
            writer.close()
            if os.path.isdir(new_dir):
                import shutil
                shutil.rmtree(new_dir, ignore_errors=True)
            os.rename(temp_dir, new_dir)
            self.dump_times += 1
        model.output_tensor_names = origin_output_tensor_names
        return acc

    def tuning_cfg_to_fw(self, tuning_cfg):
        """Parse the lpot wrapped configuration to Tensorflow.

        Args:
            tuning_cfg (dict): configuration for quantization.
        """
        self.quantize_config['calib_iteration'] = tuning_cfg['calib_iteration']
        self.quantize_config['device'] = self.device
        self.quantize_config['advance'] = deep_get(tuning_cfg, 'advance')
        fp32_ops = []
        bf16_ops = []
        int8_ops = []
        dispatched_op_names = [j[0] for j in tuning_cfg['op']]

        invalid_op_names = [i for i in self.quantize_config['op_wise_config']
                            if i not in dispatched_op_names]

        for op_name in invalid_op_names:
            self.quantize_config['op_wise_config'].pop(op_name)

        for each_op_info in tuning_cfg['op']:
            op_name = each_op_info[0]

            if tuning_cfg['op'][each_op_info]['activation']['dtype'] in ['fp32', 'bf16']:
                if op_name in self.quantize_config['op_wise_config']:
                    self.quantize_config['op_wise_config'].pop(op_name)
                if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'fp32':
                    fp32_ops.append(op_name)
                if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'bf16':
                    bf16_ops.append(op_name)
                continue

            is_perchannel = False
            weight_bit = 7.0
            if 'weight' in tuning_cfg['op'][each_op_info]:
                is_perchannel = tuning_cfg['op'][each_op_info]['weight'][
                    'granularity'] == 'per_channel'
                weight_bit = tuning_cfg['op'][each_op_info]['weight']['bit']

            algorithm = tuning_cfg['op'][each_op_info]['activation']['algorithm']

            is_asymmetric = False
            if 'activation' in tuning_cfg['op'][each_op_info]:
                is_asymmetric = tuning_cfg['op'][each_op_info]['activation']['scheme'] == 'asym'
            int8_ops.append(op_name)
            self.quantize_config['op_wise_config'][op_name] = (is_perchannel,
                                                               algorithm,
                                                               is_asymmetric,
                                                               weight_bit)
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        int8_sum_count = 0
        bf16_sum_count = 0
        log_length = 50

        if int8_ops:
            logger.info('Start to run model quantization...')

            logger.info('|' + 'Mixed Precision Statistics'.center(log_length, "*") + '|')
            for i in self._init_op_stat:
                if len(self._init_op_stat[i]) == 0:
                    continue
                count = 0
                for j in self.quantize_config['op_wise_config'].keys():
                    if j in self._init_op_stat[i]:
                        count += 1
                int8_sum_count += count
                logger.info('|' + 'INT8 {}: {} '.format(i, count).ljust(log_length) + '|')
                valid_bf16_ops = [name for name in self.bf16_ops if name in self._init_op_stat[i]]
                bf16_count = len(valid_bf16_ops)

                if bf16_count > 0:
                    logger.info('|' + 'BF16 {}: {}'.format(i, bf16_count).ljust(log_length) + '|')
                bf16_sum_count += bf16_count

            overall_ops_count = sum([len(v) for _, v in self._init_op_stat.items()])

            if overall_ops_count > 0:
                int8_percent = float(int8_sum_count / overall_ops_count)
                bf16_percent = float(bf16_sum_count / overall_ops_count)
                logger.info(('|' + 'Overall: INT8 {:.2%} ({}/{}) BF16 {:.2%} ({}/{})'.format(
                    int8_percent,
                    int8_sum_count,
                    overall_ops_count,
                    bf16_percent,
                    bf16_sum_count,
                    overall_ops_count)
                    .ljust(log_length) + '|'))
            logger.info('|' +  '*' * log_length + '|')

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            data_loader (generator): generator the data and labels
            q_func (optional): training function for quantization aware training mode,
                                which not enabled for tensorflow yet.

        Returns:
            tf.compat.v1.GraphDef: the quantized model
        """
        assert q_func is None, "quantization aware training mode is not support on tensorflow"
        self.tuning_cfg_to_fw(tune_cfg)
        logger.debug('Dump quantization configurations:')
        logger.debug(self.quantize_config)
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   qt_config=self.quantize_config,
                                   recipes=self.recipes,
                                   int8_sequences=self.op_wise_sequences,
                                   fp32_ops=self.fp32_ops,
                                   bf16_ops=self.bf16_ops,
                                   data_loader=data_loader)

        return converter.convert()

    def _query_quantizable_ops(self, matched_nodes):
        """Collect the op-wise configuration for quantization.

        Returns:
            OrderDict: op-wise configuration.
        """
        uint8_type = self.query_handler.get_op_types_by_precision(precision='uint8')
        int8_type = self.query_handler.get_op_types_by_precision(precision='int8')
        tf_quantizable_op_type = list(set(uint8_type).union(set(int8_type)))

        valid_precision = self.query_handler.get_mixed_precision_combination()
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability['uint8']['Conv2D'])
        matmul_config = copy.deepcopy(op_capability['uint8']['MatMul'])
        other_config = copy.deepcopy(op_capability['uint8']['default'])
        if ('bf16' in valid_precision and CpuInfo().bf16) or os.getenv('FORCE_BF16') == '1':
            #TODO we need to enhance below logic by introducing precision priority.
            conv_config['weight']['dtype'].insert(-1, 'bf16')
            matmul_config['weight']['dtype'].insert(-1, 'bf16')
            conv_config['activation']['dtype'].insert(-1, 'bf16')
            matmul_config['activation']['dtype'].insert(-1, 'bf16')
            other_config['activation']['dtype'].insert(-1, 'bf16')

        self.quantizable_op_details = OrderedDict()

        self._init_op_stat = {i: [] for i in tf_quantizable_op_type}

        exclude_first_quantizable_op = True if 'first_conv_or_matmul_quantization' in \
                      self.recipes and not self.recipes['first_conv_or_matmul_quantization'] \
                      else False
        for details in matched_nodes:
            node_op = details[-1][0]
            node_name = details[0]
            patterns = details[-1]
            pat_length = len(patterns)
            pattern_info = {
                'sequence': [[','.join(patterns[:pat_length - i]) for i in range(pat_length)][0]],
                'precision': ['int8']
            }
            if node_op in tf_quantizable_op_type and node_name not in self.exclude_node_names and (
                node_name, self.unify_op_type_mapping[node_op]) not in self.quantizable_op_details:
                if exclude_first_quantizable_op and \
                    (self.unify_op_type_mapping[node_op].find("conv2d") != -1 or \
                    self.unify_op_type_mapping[node_op].find("matmul") != -1):
                    exclude_first_quantizable_op = False 
                    self.exclude_node_names.append(node_name)
                    continue
                self._init_op_stat[node_op].append(node_name)
                if self.unify_op_type_mapping[node_op].find("conv2d") != -1:
                    conv2d_int8_config = copy.deepcopy(conv_config)
                    conv2d_int8_config['pattern'] = pattern_info
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = conv2d_int8_config
                elif self.unify_op_type_mapping[node_op].find("matmul") != -1:

                    matmul_int8_config = copy.deepcopy(matmul_config)
                    matmul_int8_config['pattern'] = pattern_info
                    # TODO enable the sym mode once the tf fixed the mkldequantize_op.cc bug.
                    # is_positive_input = self.pre_optimizer_handle.has_positive_input(node_name)
                    # matmul_scheme = 'sym' if is_positive_input else 'asym'
                    matmul_scheme = ['asym']
                    matmul_int8_config['activation']['scheme'] = matmul_scheme
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = matmul_int8_config
                else:
                    self.quantizable_op_details[(
                        node_name, self.unify_op_type_mapping[node_op]
                    )] = copy.deepcopy(other_config)

                self.quantize_config['op_wise_config'][node_name] = (False, "minmax", False)
        return self.quantizable_op_details

    def query_fw_capability(self, model):
        """Collect the model-wise and op-wise configuration for quantization.

        Args:
            model (tf.compat.v1.GraphDef): model definition.

        Returns:
            [dict]: model-wise & op-wise configuration for quantization.
        """
        from .tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization

        self.pre_optimizer_handle = PreOptimization(model, self.optimization)

        self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model()
        model.graph_def = self.pre_optimized_model.graph_def

        self.exclude_node_names = self.pre_optimizer_handle.get_excluded_node_names()
        patterns = self.query_handler.generate_internal_patterns()
        matched_nodes = self.pre_optimizer_handle.get_matched_nodes(patterns)
        original_graph_node_name = [i.name for i in model.graph_def.node]
        matched_nodes = sorted(matched_nodes, reverse=True, key=lambda i: (
            original_graph_node_name.index(i[0]), len(i[-1])))

        def check_match(patterns, input_pattern):
            for i in patterns:
                if input_pattern == [i for i in i.replace('+', ' ').strip().split(' ') if i]:
                    return True
            return False

        copied_matched_nodes = copy.deepcopy(matched_nodes)
        for i in copied_matched_nodes:
            if i[-1][0] in self.query_handler.get_op_types()['int8']:
                continue

            if not self.pre_optimizer_handle.has_positive_input(i[0]) and \
                not check_match(self.query_handler.get_fuse_patterns()['int8'], i[-1]):
                matched_nodes.remove(i)

        del copied_matched_nodes

        self._query_quantizable_ops(matched_nodes)
        capability = {
            'optypewise': self.get_optype_wise_ability(),
        }
        capability['opwise'] = copy.deepcopy(self.quantizable_op_details)
        logger.debug('Dump framework quantization capability:')
        logger.debug(capability)

        return capability

    def set_tensor(self, model, tensor_dict):
        from .tf_utils.graph_rewriter.graph_util import GraphAnalyzer
        g = GraphAnalyzer()
        g.graph = model.graph_def
        graph_info = g.parse_graph()

        def _get_fp32_op_name(model, tensor_name):
            is_weight = False
            is_biasadd = False
            last_node_name = None
            current_node_name = None
            for each_node in model.graph_def.node:

                if tensor_name in each_node.input:
                    tensor_index = list(each_node.input).index(tensor_name)
                    if each_node.op.find("Quantized") != -1 and tensor_index == 2:
                        is_biasadd = True
                        last_node_name = each_node.input[0]
                        current_node_name = each_node.name

                if tensor_name + "_qint8_const" in each_node.input:
                    pass

            return is_weight, is_biasadd, current_node_name, last_node_name

        from lpot.adaptor.tf_utils.graph_rewriter.graph_util import GraphRewriterHelper as Helper
        from tensorflow.python.framework import dtypes
        from tensorflow.python.framework import tensor_util
        from tensorflow.core.framework import attr_value_pb2
        qint32_type = dtypes.qint32.as_datatype_enum

        for tensor_name, tensor_content in tensor_dict.items():
            is_weight, is_biasadd, current_node_name, last_node_name = \
                    _get_fp32_op_name(model, tensor_name)

            if is_biasadd:
                is_biasadd_dtype_is_fp32 = graph_info[\
                        current_node_name].node.attr['Tbias'] == attr_value_pb2.AttrValue(
                    type=dtypes.float32.as_datatype_enum)
                current_node = graph_info[current_node_name].node
                bias_add_node = graph_info[current_node.input[2]].node
                if is_biasadd_dtype_is_fp32:
                    bias_add_node.attr["value"].CopyFrom(
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(tensor_content,
                            dtypes.float32, tensor_content.shape)))
                else:
                    last_node = graph_info[last_node_name].node
                    min_input = graph_info[\
                            last_node.input[-2]].node.attr['value'].tensor.float_val[0]
                    max_input = graph_info[\
                            last_node.input[-1]].node.attr['value'].tensor.float_val[0]
                    channel_size = tensor_content.shape[0]
                    max_filter_node = graph_info[current_node.input[6]].node
                    min_filter_node = graph_info[current_node.input[5]].node
                    if max_filter_node.attr['value'].tensor.float_val:
                        max_filter_tensor = []
                        min_filter_tensor = []
                        max_filter_tensor.append(\
                                (max_filter_node.attr['value'].tensor.float_val)[0])
                        min_filter_tensor.append(\
                                (min_filter_node.attr['value'].tensor.float_val)[0])
                    else:
                        max_filter_tensor = tensor_util.MakeNdarray(\
                                min_filter_node.attr['value'].tensor)
                        min_filter_tensor = tensor_util.MakeNdarray(\
                                min_filter_node.attr['value'].tensor)
                    activation_range = 127.0 if \
                            current_node.attr["Tinput"].type == dtypes.qint8 else 255.0
                    updated_bias = Helper.generate_int32_bias_for_conv(\
                            tensor_content, channel_size, max_input, min_input, \
                                max_filter_tensor, min_filter_tensor, activation_range)

                    bias_add_node.attr['dtype'].CopyFrom(\
                            attr_value_pb2.AttrValue(type=qint32_type))
                    bias_add_node.attr["value"].CopyFrom(\
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(updated_bias,
                            dtypes.int32, tensor_content.shape)))
                    bias_add_node.attr['value'].tensor.dtype = qint32_type
                    current_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=qint32_type))

            if is_weight:
                tmp_const_node = Helper.create_constant_node(\
                        current_node.name + '_weights_tmp',
                        tensor_content.transpose(2,3,1,0), dtypes.float32)
                min_filter_node = graph_info[current_node.input[5]].node
                per_channel = True if min_filter_node.attr['value'].tensor.tensor_shape else False
                from .tf_utils.quantize_graph.quantize_graph_common import QuantizeGraphHelper
                original_fp32_op = current_node.op.split("With")[0].split("Quantized")[-1]
                if original_fp32_op.find("Depthwise") != -1:
                    original_fp32_op = "DepthwiseConv2dNative"
                qint8_const_node, min_node, max_node = \
                        QuantizeGraphHelper.generate_quantized_weight_node(
                            original_fp32_op, tmp_const_node, per_channel)
                g.add_node(qint8_const_node, [], [current_node.name])
                g.add_node(min_node, [], [current_node.name])
                g.add_node(max_node, [], [current_node.name])
                g.replace_constant_graph_with_constant_node(qint8_const_node, tensor_name)
                g.replace_constant_graph_with_constant_node(min_node, current_node.input[5])
                g.replace_constant_graph_with_constant_node(max_node, current_node.input[6])

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[],
                       inspect_type='activation', save_to_disk=False):
        """Collect the specified tensor's output on specified iteration.

        Args:
            model (tf.compat.v1.GraphDef): model definition.
            dataloader (generator): generate the data and labels
            op_list (list, optional): the specified op names' list. Defaults to [].
            iteration_list (list, optional): the specified iteration. Defaults to [].

        Returns:
            [dict]: the key is op_name while the value is the ndarray tensor.
        """
        logger.info("Start to run inspect_tensor..")
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   qt_config=self.quantize_config,
                                   int8_sequences=self.op_wise_sequences,
                                   data_loader=dataloader)

        dump_content = converter.inspect_tensor(\
                op_list, iteration_list, self.work_dir, inspect_type)
        if save_to_disk:
            dump_dir = os.path.join(self.work_dir, 'dump_tensor')
            os.makedirs(dump_dir, exist_ok=True)
            for index, value in enumerate(dump_content['activation']):
                tmp_dict = {}
                for k, v in value.items():
                    tmp_dict[k[0]] = v
                output_path = os.path.join(dump_dir, 'activation_iter{}.npz'.format(index + 1))
                np.savez(output_path, **tmp_dict)

            if inspect_type in ('weight', 'all') and dump_content['weight']:
                np.savez(os.path.join(dump_dir, 'weight.npz'), **dump_content['weight'])

        return dump_content

    def quantize_input(self, model):
        ''' quantize the model to be able to take quantized input
            remove graph QuantizedV2 op and move its input tensor to QuantizedConv2d
            and calculate the min-max scale

            Args:
                model (tf.compat.v1.GraphDef): The model to quantize input

            Return:
                model (tf.compat.v1.GraphDef): The quantized input model
                scale (float): The scale for dataloader to generate quantized input
        '''
        scale = None
        # quantize input only support tensorflow version > 2.1.0
        import tensorflow as tf
        if tf.version.VERSION < '2.1.0':
            logger.warning('quantize input need tensorflow version > 2.1.0')
            return model, scale

        graph_def = model.as_graph_def()
        node_name_mapping = {}
        quantize_nodes = []
        for node in graph_def.node:
            node_name_mapping[node.name] = node
            if node.op == 'QuantizeV2':
                quantize_nodes.append(node)

        target_quantize_nodes = []
        for node in quantize_nodes:
            # only support Quantizev2 input op Pad and Placeholder
            if (node_name_mapping[node.input[0]].op == 'Pad' and node_name_mapping[
                    node_name_mapping[node.input[0]].input[0]].op == 'Placeholder') or \
                    node_name_mapping[node.input[0]].op == 'Placeholder':
                target_quantize_nodes.append(node)
        assert len(target_quantize_nodes) == 1, 'only support 1 QuantizeV2 from Placeholder'
        quantize_node = target_quantize_nodes[0]

        quantize_node_input = node_name_mapping[quantize_node.input[0]]
        quantize_node_outputs = [node for node in graph_def.node
                                 if quantize_node.name in node.input]

        from .tf_utils.graph_rewriter.graph_util import GraphRewriterHelper
        if quantize_node_input.op == 'Pad':
            pad_node_input = node_name_mapping[quantize_node_input.input[0]]
            assert pad_node_input.op == 'Placeholder', \
                'only support Pad between QuantizeV2 and Placeholder'
            from tensorflow.python.framework import tensor_util
            paddings_tensor = tensor_util.MakeNdarray(node_name_mapping[
                quantize_node_input.input[1]].attr['value'].tensor).flatten()

            quantize_node.input[0] = quantize_node_input.input[0]
            for conv_node in quantize_node_outputs:
                assert 'Conv2D' in conv_node.op, 'only support QuantizeV2 to Conv2D'

                GraphRewriterHelper.set_attr_int_list(conv_node,
                                                      "padding_list", paddings_tensor)
            graph_def.node.remove(quantize_node_input)

        from tensorflow.python.framework import dtypes
        GraphRewriterHelper.set_attr_dtype(node_name_mapping[quantize_node.input[0]],
                                           "dtype", dtypes.qint8)

        for conv_node in quantize_node_outputs:
            for index, conv_input in enumerate(conv_node.input):
                if conv_input == quantize_node.name:
                    conv_node.input[index] = quantize_node.input[0]
                elif conv_input == quantize_node.name + ":1":
                    conv_node.input[index] = quantize_node.input[1]
                elif conv_input == quantize_node.name + ":2":
                    conv_node.input[index] = quantize_node.input[2]

        # get the input's min-max value and calculate scale
        max_node = node_name_mapping[quantize_node.input[2]]
        min_node = node_name_mapping[quantize_node.input[1]]
        max_value = max_node.attr['value'].tensor.float_val[0]
        min_value = min_node.attr['value'].tensor.float_val[0]
        scale = 127. / max(abs(max_value), abs(min_value))
        # remove QuantizeV2 node
        graph_def.node.remove(quantize_node)

        graph = tensorflow.Graph()
        with graph.as_default():
            # use name='' to avoid 'import/' to name scope
            tensorflow.import_graph_def(graph_def, name='')
        return graph, scale

    def get_optype_wise_ability(self):
        """Get the op type wise capability by generating the union value of each op type.
        Returns:
            [string dict]: the key is op type while the value is the
                           detail configurations of activation and weight for this op type.
        """
        res = OrderedDict()
        for op in self.quantizable_op_details:
            if op[1] not in res:
                res[op[1]] = {'activation': self.quantizable_op_details[op]['activation']}
                if 'weight' in self.quantizable_op_details[op]:
                    res[op[1]]['weight'] = self.quantizable_op_details[op]['weight']
        return res

    def _pre_eval_hook(self, model):
        return model

    def _post_eval_hook(self, model):
        pass

    def save(self, model, path):
        pass

    def convert(self, model, source, destination):
        '''The function is used to convert a source model format to another.

           Args:
               model (lpot.model): base model to be converted.
               source (string): The source model format.
               destination (string): The destination model format.
        '''
        assert source.lower() == 'qat' and destination.lower() == 'default'
        capability = self.query_fw_capability(model)
        print(capability['opwise'])
        quantize_config = {'op_wise_config': {}}
        for each_op_info in capability['opwise']: 
            op_name = each_op_info[0]
            op_type = each_op_info[1]

            is_perchannel = False
            weight_bit = 7.0
            activation = capability['optypewise'][op_type]['activation']
            if 'weight' in capability['optypewise'][op_type]:
                weight = capability['optypewise'][op_type]['weight']
                is_perchannel = True if weight[
                    'granularity'][0] == 'per_channel' else False

            algorithm = activation['algorithm'][0]

            is_asymmetric = False
            if 'activation' in capability['optypewise'][op_type]:
                is_asymmetric = True if activation['scheme'][0] == 'asym' else False

            quantize_config['op_wise_config'][op_name] = (is_perchannel,
                                                          algorithm,
                                                          is_asymmetric,
                                                          weight_bit)
        from .tf_utils.graph_converter import GraphConverter
        converter = GraphConverter(model,
                                   qt_config=quantize_config,
                                   int8_sequences=self.op_wise_sequences,
                                   fake_quant=True)

        return converter.convert()


@singleton
class TensorflowQuery(QueryBackendCapability):

    def __init__(self, local_config_file=None):
        import tensorflow as tf

        super().__init__()
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

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

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                self.logger.info("Failed to parse {} due to {}".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check the {} format.".format(self.cfg))

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

    def get_fuse_patterns(self):
        """Get supported patterns by low precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the supported patterns.
        """
        return self.cur_config['patterns']

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

    def get_mixed_precision_combination(self):
        """Get the valid mixed precisions.

        Returns:
            [string list]: valid precision list.
        """
        if self.cur_config['precisions']['valid_mixed_precisions']:
            return [i.strip() for i in self.cur_config['precisions']['valid_mixed_precisions']]

        return [i.strip() for i in self.get_precisions().split(',')]

    def get_grappler_optimization_cfg(self):
        return self.cur_config['grappler_optimization']

    def get_eightbit_patterns(self):
        """Get eightbit op wise sequences information.

        Returns:
            [dictionary]: key is the op type while value is the list of sequences start
                        with the op type same as key value.
        """
        quantizable_op_types = self.get_op_types_by_precision(
            'int8') + self.get_op_types_by_precision('uint8')
        int8_patterns = [i.replace(
            '+', ' ').split() for i in list(set(self.get_fuse_patterns()['int8'] +
                                                self.get_fuse_patterns()['uint8']))]

        res = {}
        for i in quantizable_op_types:
            res[i] = [[i]]

        for pattern in int8_patterns:
            op_type = pattern[0]
            if op_type in res:
                res[op_type].append(pattern)

        return res

    def generate_internal_patterns(self):
        """Translate the patterns defined in the yaml to internal pattern expression.
        """
        def _generate_pattern(data):
            length = [len(i) for i in data]
            res=[]
            for index in range(max(length)):
                if index <= min(length) - 1:
                    tmp = [i[index] for i in data]
                    if len(set(tmp)) == 1:
                        res.append([tmp[0]])
                    else:
                        res.append(tuple(set(tmp)))
                else:
                    tmp1 = [i[index] for i in data if len(i) > index]
                    res.append(tuple(set(tmp1)))

            return res

        op_level_sequences = {}

        for k, op_level_all_sequences in self.get_eightbit_patterns().items():
            op_level_sequences[k] = []
            sorted_sequences = sorted(op_level_all_sequences)
            last_len = 1
            each_combination = []
            for index, value in enumerate(sorted_sequences):
                if  len(value) >= last_len:
                    last_len = len(value)
                    each_combination.append(value)
                else:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))
                    each_combination.clear()
                    each_combination.append(value)
                    last_len = len(value)

                if index == len(sorted_sequences) - 1:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))

        final_out = []
        for _ , op_level_sequences in op_level_sequences.items():
            for similar_sequences in op_level_sequences:
                final_out.append(_generate_pattern(similar_sequences))

        return final_out

#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from lpot.utils.utility import get_all_fp32_data
from lpot.utils.utility import get_tensor_histogram
from lpot.utils.utility import combine_histogram
from lpot.utils.utility import CaptureOutputToFile
from lpot.utils.utility import str2array
from lpot.conf.dotdict import deep_get
from .transform_graph.insert_logging import InsertLogging
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .transform_graph.bias_correction import BiasCorrection
from .util import write_graph
from .util import get_graph_def
from .util import get_tensor_by_name, iterator_sess_run
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph.quantize_graph_common import QuantizeGraphHelper
from .quantize_graph.quantize_graph_conv import FuseNodeStartWithConv2d

from .graph_rewriter.generic.remove_training_nodes import RemoveTrainingNodesOptimizer
from .graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from .graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer

from .graph_rewriter.int8.freeze_value import FreezeValueTransformer
from .graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeDequantizeTransformer
from .graph_rewriter.int8.insert_logging import InsertLoggingTransformer
from .graph_rewriter.int8.scale_propagation import ScaleProPagationTransformer
from .graph_rewriter.bf16.bf16_convert import BF16Convert
from .graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
TF_SUPPORTED_MAX_VERSION = '2.3.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'

class GraphConverter:
    def __init__(self,
                 input_graph,
                 output_graph,
                 inputs=[],
                 outputs=[],
                 qt_config={},
                 int8_sequences={},
                 fp32_ops=[],
                 bf16_ops=[],
                 data_loader=None):
        """Convert graph.

        :param input_graph: input graph pb file.
        :param output_graph: output graph pb file. If set, output directory should be exist.
        :param inputs: input nodes' names.
        :param outputs: output nodes' names.
        :param qt_config: quantization configs, including interation and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        """
        # Logger initial
        self.logger = logging.getLogger()
        self.debug = bool(self.logger.level == logging.DEBUG)

        # as we may have outputs with suffix, strip to get raw name
        self.output_node_names = list(set([x.split(":")[0] for x in outputs]))
        self.input_node_names = list(set([x.split(":")[0] for x in inputs]))
        # For lpot, the input_graph is not graph file path but Graph object.
        self.input_graph = get_graph_def(input_graph, self.output_node_names)
        if 'MakeIterator' in [node.op for node in self.input_graph.node]:
            self.output_node_names.append('MakeIterator')
        self.output_graph = output_graph
        self.input_tensor_names = inputs
        self.output_tensor_names = outputs

        # quantize specific config
        self.calib_iteration = qt_config['calib_iteration']
        self.op_wise_config = qt_config['op_wise_config']
        self.advance_config = deep_get(qt_config, 'advance')
        self.device = qt_config['device'] if 'device' in qt_config else 'cpu'
        self.int8_sequences = int8_sequences
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops

        self._calibration_data = []
        self._fp32_print_data = []
        self.data_loader = data_loader
        self._check_tf_version()
        self._check_args()
        self._gen_tmp_filenames()
        self._kl_op_dict = {}
        self._kl_keys = []
        self._print_node_mapping = {}
        self._enable_kl_op_names = [
            k for k in self.op_wise_config if self.op_wise_config[k][1] == 'kl'
        ]
        self._fp32_origin_graph = copy.deepcopy(self.input_graph)

    def _inference(self, input_graph):
        """Run the calibration on the input graph

        Args:
            input_graph (tf.compat.v1.GraphDef): input graph
        """
        import tensorflow as tf

        graph = tf.Graph()
        graph_def = get_graph_def(input_graph)
        assert graph_def
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        iter_op = None
        if 'MakeIterator' in self.output_node_names:
            iter_op = graph.get_operation_by_name('MakeIterator')

        input_tensor = [get_tensor_by_name(graph, x) for x in self.input_tensor_names]
        output_tensor = [
            get_tensor_by_name(graph, x) for x in self.output_tensor_names \
            ] if len(self.output_tensor_names) > 1 else \
            get_tensor_by_name(graph, self.output_tensor_names[0])

        config = tf.compat.v1.ConfigProto()
        # config.use_per_session_threads = 1
        config.inter_op_parallelism_threads = 1
        sess = tf.compat.v1.Session(graph=graph, config=config)

        self.logger.info("Sampling data...")
        for idx, (inputs, labels) in enumerate(self.data_loader):
            if len(input_tensor) == 1:
                feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), \
                    'inputs len must equal with input_tensor'
                feed_dict = dict(zip(input_tensor, inputs))

            _ = sess.run(output_tensor, feed_dict) if iter_op is None else \
                iterator_sess_run(sess, iter_op, feed_dict, output_tensor, self.calib_iteration)

            if idx + 1 == self.calib_iteration:
                break

        sess.close()

    def _check_tf_version(self):
        is_supported_version = False
        try:
            from tensorflow import python
            if (hasattr(python, "pywrap_tensorflow")
                    and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (TF_SUPPORTED_MIN_VERSION <= tf.version.VERSION):
                is_supported_version = True
        except Exception as e:
            raise ValueError(e)
        finally:
            if tf.version.VERSION > TF_SUPPORTED_MAX_VERSION:
                self.logger.warning(
                    str('Please note the {} version of Intel® Optimizations for'
                        ' TensorFlow is not fully verified!'
                        ' Suggest to use the versions'
                        ' between {} and {} if meet problem').format(tf.version.VERSION,
                                                                     TF_SUPPORTED_MIN_VERSION,
                                                                     TF_SUPPORTED_MAX_VERSION))
            if not is_supported_version:
                raise ValueError(
                    str('Please install Intel® Optimizations for TensorFlow'
                        ' or MKL enabled source build TensorFlow'
                        ' with version >={} and <={}').format(TF_SUPPORTED_MIN_VERSION,
                                                              TF_SUPPORTED_MAX_VERSION))

    def _check_args(self):
        if self.output_graph and not os.path.exists(os.path.dirname(self.output_graph)):
            raise ValueError('"output_graph" directory does not exist.')

        self._output_path = os.path.dirname(
            os.path.realpath(self.output_graph if self.output_graph else self.input_graph))

    def _gen_tmp_filenames(self):
        self._fp32_optimized_graph = os.path.join(self._output_path, 'fp32_optimized_graph.pb')
        self._int8_dynamic_range_graph = os.path.join(self._output_path,
                                                      'int8_dynamic_range_graph.pb')
        self._int8_logged_graph = os.path.join(self._output_path, 'int8_logged_graph.pb')
        self._fp32_logged_graph = os.path.join(self._output_path, 'fp32_logged_graph.pb')
        self._int8_frozen_range_graph = os.path.join(self._output_path,
                                                     'int8_frozen_range_graph.pb')
        self._bf16_mixed_precision_graph = os.path.join(self._output_path,
                                                        'int8_bf16_mixed_precision_graph.pb')
        if not self.output_graph:
            self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph.pb')
        # to keep temp graphDef
        self._tmp_graph_def = copy.deepcopy(self.input_graph)

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.
            5) bf16 convert if the self.bf16_ops is not empty

        :return:
        """
        try:
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(self._tmp_graph_def, name='')
        except Exception as e:
            self.logger.error('Failed to optimize fp32 graph due to: %s', str(e))
            raise ValueError(e) from e
        else:
            if len(self.op_wise_config) > 0:
                graph = self.quantize()
            if len(self.bf16_ops) > 0:
                graph = self.bf16_convert()
            return graph

    def _get_fp32_print_node_names(self, specified_op_list):
        offset_map = {
            "QuantizedConv2DWithBiasSumAndRelu": 3,
            "QuantizedConv2DWithBiasAndRelu": 2,
            "QuantizedConv2DWithBias": 1,
        }
        target_conv_op = []
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(self._fp32_origin_graph,
                                                              self.input_node_names,
                                                              self.output_node_names)

        node_name_mapping = {
            node.name: node
            for node in self._tmp_graph_def.node if node.op != "Const"
        }

        for node in self._tmp_graph_def.node:
            if node.op in offset_map:
                target_conv_op.append(node.name.split('_eightbit_')[0])
        fp32_node_name_mapping = {
            node.name: node
            for node in sorted_graph.node if node.op != "Const"
        }
        sorted_node_names = [i.name for i in sorted_graph.node if i.op != "Const"]

        output_node_names = []
        for i in target_conv_op:
            if specified_op_list and i not in specified_op_list:
                continue
            if node_name_mapping[i + "_eightbit_quantized_conv"].op == \
                    'QuantizedConv2DWithBiasSumAndRelu':
                start_index = sorted_node_names.index(i)
                for index, value in enumerate(sorted_node_names[start_index:]):
                    if fp32_node_name_mapping[value].op.startswith(
                            "Add") and fp32_node_name_mapping[sorted_node_names[start_index +
                                                                                index +
                                                                                1]].op == "Relu":
                        output_node_names.append(sorted_node_names[start_index + index + 1])
                        self._print_node_mapping[sorted_node_names[start_index + index + 1]] = i

            elif i in sorted_node_names:
                start_index = sorted_node_names.index(i)
                end_index = start_index + offset_map[node_name_mapping[
                    i + "_eightbit_quantized_conv"].op]
                output_node_names.append(sorted_node_names[end_index])
                self._print_node_mapping[sorted_node_names[end_index]] = i

        for i in output_node_names:
            self._kl_keys.append(';' + i + '__print__;__KL')

        InsertLogging(self._fp32_origin_graph,
                      node_name_list=output_node_names,
                      message="__KL:",
                      summarize=-1,
                      dump_fp32=True).do_transformation()

        write_graph(self._fp32_origin_graph, self._fp32_logged_graph)
        return self._fp32_origin_graph

    def _dequantize(self, data, scale_info):
        original_shape = data.shape
        size = data.size
        new_data = data.reshape(size, )
        max_value = 255 if scale_info[0].find("Relu") != -1 else 127
        return np.array([float(i / max_value) for i in new_data]).reshape(original_shape)

    def inspect_tensor(self, original_op_list, iteration_list, work_dir):
        """dump the specified op's output tensor content

        Args:
            original_op_list (string list): the ops name
            iteration_list (int list): the specified iteration to dump tensor

        Returns:
            dict: key is op name while value is the content saved in np.array format.
        """
        graph_node_name_mapping = {}
        q_node_name = []
        fp32_node_name = []
        fp32_node_name_mapping = {}
        q_node_scale = {}
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(self._fp32_origin_graph,
                                                              self.input_node_names,
                                                              self.output_node_names)
        graph_q_node_name = []
        op_name_type_dict = {}
        quantized_node_name_postfix = '_eightbit_requantize'
        for node in sorted_graph.node:
            node_name = node.name
            if node.op.find("Quantized") != -1:
                node_name = node.name.split(quantized_node_name_postfix)[0]
                graph_q_node_name.append(node_name)
            graph_node_name_mapping[node_name] = node

        for op_info in original_op_list:
            op_name = op_info[0]
            op_type = op_info[1]

            if op_type not in ["conv2d"]:
                continue
            op_name_type_dict[op_name] = op_type

            if op_name in graph_q_node_name:
                q_node_name.append(op_name + quantized_node_name_postfix)
                q_node = graph_node_name_mapping[op_name]
                q_out_min = graph_node_name_mapping[
                    q_node.input[-2]].attr["value"].tensor.float_val[0]
                q_out_max = graph_node_name_mapping[
                    q_node.input[-1]].attr["value"].tensor.float_val[0]
                q_node_scale[op_name + quantized_node_name_postfix] = (q_node.op, q_out_min,
                                                                       q_out_max)
            else:
                fp32_node_name.append(op_name)
                node_op =  graph_node_name_mapping[op_name].op
                if node_op in ("Conv2D", "DepthwiseConv2dNative"):
                    _, matched_nodes = FuseNodeStartWithConv2d(
                        input_graph=sorted_graph,
                        patterns=self.int8_sequences[node_op],
                        remove_redundant_quant_flag=True,
                        op_wise_cfg=(False, "minmax", False),
                        start_node_name=op_name,
                        device=self.device).get_longest_fuse()

                    if matched_nodes:
                        fp32_node_name_mapping[matched_nodes[-1]] = op_name
                else:
                    fp32_node_name_mapping[op_name] = op_name

        InsertLogging(sorted_graph,
                      node_name_list=fp32_node_name_mapping.keys(),
                      message="__KL:",
                      summarize=-1,
                      dump_fp32=True).do_transformation()

        if q_node_name:
            sorted_graph = InsertLogging(sorted_graph,
                                         node_name_list=q_node_name,
                                         message="__KL:",
                                         summarize=-1).do_transformation()

        tmp_dump_file = os.path.join(work_dir, 'kl.log')
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(sorted_graph)

        with open(tmp_dump_file) as f:
            disk_content = f.readlines()

        filter_content = (i for i in disk_content if i.startswith(';'))

        dump_tensor_content = {}

        for i in filter_content:
            contents = i.split('__print__;__KL:')
            node_name = contents[0][1:]
            node_content = str2array(contents[1])

            if node_name not in dump_tensor_content:
                dump_tensor_content[node_name] = []
            dump_tensor_content[node_name].append(node_content)

        result_disk = {}
        tensor_iter_idx = iteration_list[0] - 1 if iteration_list else 0
        for k, v in dump_tensor_content.items():
            if k in fp32_node_name_mapping:
                key = fp32_node_name_mapping[k]
                result_disk[(key, op_name_type_dict[key])] = v[tensor_iter_idx]
            else:
                result_key = k.split(quantized_node_name_postfix)[tensor_iter_idx]
                result_disk[(result_key, op_name_type_dict[result_key])
                            ] = self._dequantize(v[0], q_node_scale[k])
        return result_disk

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            self._quantize_graph()
            if self._enable_kl_op_names:
                self._get_fp32_print_node_names(self._enable_kl_op_names)
                self._generate_calibration_data(self._fp32_logged_graph, self._fp32_print_data,
                                                True)
            self._insert_logging()
            self._generate_calibration_data(self._int8_logged_graph, self._calibration_data)

            if len(self._calibration_data) > 0:
                self._freeze_requantization_ranges(self._kl_op_dict)
                self._fuse_requantize_with_fused_quantized_node()
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(self._tmp_graph_def, name='')
        except Exception as e:
            import traceback
            traceback.print_exc()
            graph = None
            self.logger.error('Failed to quantize graph due to: %s', str(e))
        finally:
            if not self.debug:
                self._post_clean()
            return graph

    def bf16_convert(self):
        """Convert fp32 nodes in bf16_node to bf16 dtype based on
           FP32 + INT8 mixed precision graph.
        """
        try:
            self._tmp_graph_def = BF16Convert(self._tmp_graph_def, self.fp32_ops,
                                              self.bf16_ops).do_transformation()
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(self._tmp_graph_def, name='')
        except Exception as e:
            graph = None
            self.logger.error('Failed to convert graph due to: %s', str(e))
        finally:
            if self.debug:
                write_graph(self._tmp_graph_def, self._bf16_mixed_precision_graph)
            return graph

    def _quantize_graph(self):
        """quantize graph."""

        g = ops.Graph()
        with g.as_default():
            importer.import_graph_def(self._tmp_graph_def)
        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))

        self._tmp_graph_def = FusePadWithConv2DOptimizer(self._tmp_graph_def,
                                                         non_pad_ops,
                                                         self.input_node_names,
                                                         self.op_wise_config).do_transformation()

        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(self._tmp_graph_def,
                                                                     self.input_node_names,
                                                                     self.output_node_names)
        intel_quantizer = QuantizeGraphForIntel(self._tmp_graph_def,
                                                self.output_node_names,
                                                self.op_wise_config,
                                                self.int8_sequences,
                                                self.device)

        self._tmp_graph_def = intel_quantizer.do_transform()

        self._tmp_graph_def.library.CopyFrom(self.input_graph.library)

        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_dynamic_range_graph)

    def _insert_logging(self):
        int8_dynamic_range_graph_def = graph_pb2.GraphDef()
        int8_dynamic_range_graph_def.CopyFrom(self._tmp_graph_def)
        # TODO need to insert op-wise logging op.
        self._tmp_graph_def = InsertLoggingTransformer(self._tmp_graph_def,
                                                       target_op_types=[
                                                        "RequantizationRange",
                                                        "RequantizationRangePerChannel"],
                                                       message="__requant_min_max:"). \
                                                       do_transformation()

        self._tmp_graph_def = InsertLoggingTransformer(
            self._tmp_graph_def, target_op_types=["Min"], message="__min:").do_transformation()

        self._tmp_graph_def = InsertLoggingTransformer(
            self._tmp_graph_def, target_op_types=["Max"], message="__max:").do_transformation()
        self._tmp_graph_def.library.CopyFrom(self.input_graph.library)
        write_graph(self._tmp_graph_def, self._int8_logged_graph)
        self._tmp_graph_def.CopyFrom(int8_dynamic_range_graph_def)

    def _generate_calibration_data(self, graph, output_data, enable_kl_algo=False):

        tmp_dump_file = os.path.join(os.path.dirname(self.output_graph), 'requant_min_max.log')

        self.logger.debug("Generating calibration data and saving to {}".format(tmp_dump_file))

        with CaptureOutputToFile(tmp_dump_file):
            self._inference(graph)

        with open(tmp_dump_file) as f:
            output_data.extend(f.readlines())

        for line in output_data:
            if enable_kl_algo and line.rsplit(':')[0] in self._kl_keys:
                fp32_data = get_all_fp32_data(line.rsplit(':')[-1])
                key = self._print_node_mapping[line[1:].split('__print')
                                               [0]] + '_eightbit_requant_range'
                if key not in self._kl_op_dict:
                    self._kl_op_dict[key] = get_tensor_histogram(fp32_data)
                else:
                    self._kl_op_dict[key] = combine_histogram(self._kl_op_dict[key], fp32_data)

    def _freeze_requantization_ranges(self, additional_data=None):
        self._tmp_graph_def = FreezeValueTransformer(self._tmp_graph_def, self._calibration_data,
                                                     '__max:').do_transformation()

        self._tmp_graph_def = FreezeValueTransformer(self._tmp_graph_def, self._calibration_data,
                                                     '__min:').do_transformation()

        self._tmp_graph_def = FreezeValueTransformer(self._tmp_graph_def,
                                                     self._calibration_data,
                                                     '__requant_min_max',
                                                     tensor_data= additional_data,
                                                     device=self.device,
                                                     ).do_transformation()
        self._tmp_graph_def = ScaleProPagationTransformer(self._tmp_graph_def).do_transformation()
        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_frozen_range_graph)

    def _fuse_requantize_with_fused_quantized_node(self):
        self._tmp_graph_def = FuseConvRequantizeTransformer(self._tmp_graph_def,
                                                            self.device).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeDequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(self._tmp_graph_def,
                                                        self.input_node_names,
                                                        self.output_node_names).do_transformation()

        self._tmp_graph_def = RemoveTrainingNodesOptimizer(self._tmp_graph_def,
                                                           protected_nodes=self.output_node_names
                                                          ).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = RerangeQuantizedConcat(self._tmp_graph_def,
                                                     self.device).do_transformation()

        self._tmp_graph_def = PostCseOptimizer(self._tmp_graph_def).do_transformation()

        if self.advance_config is not None and \
           deep_get(self.advance_config, 'bias_correction') is not None:
            self._tmp_graph_def = BiasCorrection(
                self._tmp_graph_def, self.input_graph).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.input_graph.library)

        if self.debug:
            write_graph(self._tmp_graph_def, self.output_graph)
            self.logger.info('Converted graph file is saved to: %s', self.output_graph)

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if gfile.Exists(self._int8_logged_graph):
            os.remove(self._int8_logged_graph)

#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2021 Intel Corporation
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
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from lpot.utils.utility import get_all_fp32_data
from lpot.utils.utility import get_tensor_histogram
from lpot.utils.utility import combine_histogram
from lpot.utils.utility import CaptureOutputToFile
from lpot.utils.utility import str2array
from lpot.utils.utility import Dequantize
from lpot.conf.dotdict import deep_get
from lpot.model.model import TensorflowModel
from .transform_graph.insert_logging import InsertLogging
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .transform_graph.bias_correction import BiasCorrection
from .util import iterator_sess_run
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
from .graph_rewriter.int8.meta_op_optimizer import MetaInfoChangingMemOpOptimizer

TF_SUPPORTED_MAX_VERSION = '2.4.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'

class GraphConverter:
    def __init__(self,
                 model,
                 qt_config={},
                 recipes={},
                 int8_sequences={},
                 fp32_ops=[],
                 bf16_ops=[],
                 data_loader=None):
        """Convert graph.

        :param model: input tensorflow model.
        :param qt_config: quantization configs, including interation and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        """
        # Logger initial
        self.logger = logging.getLogger()
        self.debug = bool(self.logger.level == logging.DEBUG)
        self.model = model
        # quantize specific config
        self.calib_iteration = qt_config['calib_iteration']
        self.op_wise_config = qt_config['op_wise_config']
        self.advance_config = deep_get(qt_config, 'advance')
        self.device = qt_config['device'] if 'device' in qt_config else 'cpu'
        self.int8_sequences = int8_sequences
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.recipes = recipes

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
        self._fp32_model = TensorflowModel(self.model.model,
                                           self.model.framework_specific_info,
                                           **self.model.kwargs)
        self._tmp_graph_def = copy.deepcopy(self.model.graph_def)
    # pylint: disable=no-member 
    def _inference(self, model):
        """Run the calibration on the input graph

        Args:
            model(TensorflowModel): input TensorflowModel
        """
        input_tensor = model.input_tensor
        output_tensor = model.output_tensor

        self.logger.info("Sampling data...")
        for idx, (inputs, labels) in enumerate(self.data_loader):
            if len(input_tensor) == 1:
                feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), \
                    'inputs len must equal with input_tensor'
                feed_dict = dict(zip(input_tensor, inputs))

            _ = model.sess.run(output_tensor, feed_dict) if model.iter_op is None \
                else iterator_sess_run(model.sess, model.iter_op, \
                    feed_dict, output_tensor, self.calib_iteration)

            if idx + 1 == self.calib_iteration:
                break

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
        if self.model.workspace_path and not os.path.isdir(self.model.workspace_path) \
                and not os.path.exists(os.path.dirname(self.model.workspace_path)):
            raise ValueError('"output_graph" directory does not exist.')
        self._output_path = self.model.workspace_path

    def _gen_tmp_filenames(self):
        self._int8_dynamic_range_model_path = os.path.join(self._output_path, \
                                                      'int8_dynamic_range_graph')
        self._int8_logged_model_path = os.path.join(self._output_path, 'int8_logged_graph')
        self._fp32_logged_model_path = os.path.join(self._output_path, 'fp32_logged_graph')
        self._int8_frozen_range_model_path = os.path.join(self._output_path,
                                                          'int8_frozen_range_graph')
        self._bf16_mixed_precision_model_path = os.path.join(self._output_path,
                                                        'int8_bf16_mixed_precision_graph')

        self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph')
        # to keep temp model
        self._tmp_model = TensorflowModel(self.model.model, \
                                          self.model.framework_specific_info,
                                          **self.model.kwargs)

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.
            5) bf16 convert if the self.bf16_ops is not empty

        :return:
        """
        model = self._tmp_model

        if len(self.op_wise_config) > 0:
            model = self.quantize()

        if len(self.bf16_ops) > 0:
            model = self.bf16_convert()

        post_cse_graph_def = PostCseOptimizer(model.graph_def).do_transformation()
        post_cse_graph_def.library.CopyFrom(self.model.graph_def.library)
        model.graph_def = post_cse_graph_def

        if self.debug:
            model.save(self.output_graph)
            self.logger.info('Converted graph file is saved to: %s', self.output_graph)

        return model

    def _get_fp32_print_node_names(self, specified_op_list):
        offset_map = {
            "QuantizedConv2DWithBiasSumAndRelu": 3,
            "QuantizedConv2DWithBiasAndRelu": 2,
            "QuantizedConv2DWithBias": 1,
        }
        target_conv_op = []
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_model.graph_def,
            self._fp32_model.input_node_names,
            self._fp32_model.output_node_names)

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

        fp32_graph_def = graph_pb2.GraphDef()
        fp32_graph_def.CopyFrom(self._fp32_model.graph_def)
        self._fp32_model.graph_def = InsertLogging(self._fp32_model.graph_def,
                      node_name_list=output_node_names,
                      message="__KL:",
                      summarize=-1,
                      dump_fp32=True).do_transformation()

        self._fp32_model.save(self._fp32_logged_model_path)
        self._fp32_model.graph_def = fp32_graph_def
        return self._fp32_model

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
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_model.graph_def,
            self._fp32_model.input_node_names,
            self._fp32_model.output_node_names)

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

        model = TensorflowModel(sorted_graph, self._tmp_model.framework_specific_info)
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(model)

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
                            ] = Dequantize(v[0], q_node_scale[k])
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
                self._generate_calibration_data(self._fp32_logged_model_path,
                                                self._fp32_print_data,
                                                True)

            self._insert_logging()
            self._generate_calibration_data(self._int8_logged_model_path,
                                            self._calibration_data)
            if len(self._calibration_data) > 0:
                self._freeze_requantization_ranges(self._kl_op_dict)
                self._fuse_requantize_with_fused_quantized_node()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._tmp_model = None
            self.logger.error('Failed to quantize graph due to: %s', str(e))
        finally:
            if not self.debug:
                self._post_clean()
            return self._tmp_model

    def bf16_convert(self):
        """Convert fp32 nodes in bf16_node to bf16 dtype based on
           FP32 + INT8 mixed precision graph.
        """
        try:
            self._tmp_model.graph_def = BF16Convert(
                self._tmp_model.graph_def,
                self.fp32_ops,
                self.bf16_ops).do_transformation()

        except Exception as e:
            self._tmp_model = None
            self.logger.error('Failed to convert graph due to: %s', str(e))
        finally:
            if self.debug:
                self._tmp_model.save(self._bf16_mixed_precision_model_path)

            return self._tmp_model

    def _quantize_graph(self):
        """quantize graph."""

        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))

        self._tmp_graph_def = FusePadWithConv2DOptimizer(
            self._tmp_graph_def,
            non_pad_ops,
            self._tmp_model.input_node_names,
            self.op_wise_config).do_transformation()

        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names)

        self._tmp_graph_def = QuantizeGraphForIntel(
            self._tmp_graph_def,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device).do_transform()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        if self.debug:
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_dynamic_range_model_path)

    def _insert_logging(self):
        int8_dynamic_range_graph_def = graph_pb2.GraphDef()
        int8_dynamic_range_graph_def.CopyFrom(self._tmp_graph_def)
        # TODO need to insert op-wise logging op.
        self._tmp_graph_def = InsertLoggingTransformer(
            self._tmp_graph_def,
            target_op_types=[
             "RequantizationRange",
             "RequantizationRangePerChannel"],
             message="__requant_min_max:").do_transformation()

        self._tmp_graph_def = InsertLoggingTransformer(
            self._tmp_graph_def,
            target_op_types=["Min"],
            message="__min:").do_transformation()

        self._tmp_graph_def = InsertLoggingTransformer(
            self._tmp_graph_def,
            target_op_types=["Max"],
            message="__max:").do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def
        self._tmp_model.save(self._int8_logged_model_path)

        self._tmp_graph_def.CopyFrom(int8_dynamic_range_graph_def)

    def _generate_calibration_data(self, tmp_path, output_data, enable_kl_algo=False):

        tmp_dump_file = os.path.join(os.path.dirname(self.output_graph), 'requant_min_max.log')

        self.logger.debug("Generating calibration data and saving to {}".format(tmp_dump_file))

        model = TensorflowModel(tmp_path,
                                self._tmp_model.framework_specific_info,
                                **self._tmp_model.kwargs)

        with CaptureOutputToFile(tmp_dump_file):
            self._inference(model)

        with open(tmp_dump_file, errors='ignore') as f:
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
        self._tmp_graph_def = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__max:').do_transformation()

        self._tmp_graph_def = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__min:').do_transformation()

        self._tmp_graph_def = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__requant_min_max',
            tensor_data= additional_data,
            device=self.device,
            ).do_transformation()

        if self.recipes['scale_propagation_max_pooling']:
            self._tmp_graph_def = ScaleProPagationTransformer(
                self._tmp_graph_def).do_transformation()

        if self.debug:
            self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_frozen_range_model_path)

    def _fuse_requantize_with_fused_quantized_node(self):
        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def,
            self.device).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseMatMulRequantizeDequantizeTransformer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names).do_transformation()

        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def,
            protected_nodes=self._tmp_model.output_node_names).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(
            self._tmp_graph_def).do_transformation()

        if self.recipes['scale_propagation_concat']:
            self._tmp_graph_def = RerangeQuantizedConcat(self._tmp_graph_def,
                                                     self.device).do_transformation()

        self._tmp_graph_def = MetaInfoChangingMemOpOptimizer(
            self._tmp_graph_def).do_transformation()

        if self.advance_config is not None and \
           deep_get(self.advance_config, 'bias_correction') is not None:
            self._tmp_graph_def = BiasCorrection(
                self._tmp_graph_def, self.model.graph_def).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if os.path.exists(self._int8_logged_model_path) and \
            os.path.isdir(self._int8_logged_model_path):
            import shutil
            shutil.rmtree(self._int8_logged_model_path)

        elif gfile.Exists(self._int8_logged_model_path + '.pb'):
            os.remove(self._int8_logged_model_path + '.pb')

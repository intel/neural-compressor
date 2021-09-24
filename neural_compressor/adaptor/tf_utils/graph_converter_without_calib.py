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
from tensorflow.python.platform import gfile
from neural_compressor.conf.dotdict import deep_get
from neural_compressor.experimental.common import Model
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .transform_graph.bias_correction import BiasCorrection
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph.quantize_graph_common import QuantizeGraphHelper

from .graph_rewriter.graph_util import GraphAnalyzer
from .graph_rewriter.generic.remove_training_nodes import RemoveTrainingNodesOptimizer
from .graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from .graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer

from .graph_rewriter.int8.freeze_value_without_calib import FreezeValueWithoutCalibTransformer
from .graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeDequantizeTransformer
from .graph_rewriter.int8.scale_propagation import ScaleProPagationTransformer
from .graph_rewriter.bf16.bf16_convert import BF16Convert
from .graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
from .graph_rewriter.int8.meta_op_optimizer import MetaInfoChangingMemOpOptimizer
from .graph_rewriter.int8.rnn_convert import QuantizedRNNConverter


TF_SUPPORTED_MAX_VERSION = '2.6.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'

logger = logging.getLogger()
debug = bool(logger.level == logging.DEBUG)

class GraphConverterWithoutCalib:
    def __init__(self,
                 model,
                 data_loader=None,
                 recover_config=None):
        """Convert graph without calibration.

        :param model: input tensorflow model.
        :param qt_config: quantization configs, including interation and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        :param recover_config: config for recovering tuned model
        """
        # Logger initial
        self.model = model
        #(TODO) does it right to make the internal model format as graph_def
        self.output_tensor_names = self.model.output_tensor_names
        self.input_tensor_names = self.model.input_tensor_names
        # quantize specific config
        self.op_wise_config = recover_config['op_wise_config']
        self.advance_config = deep_get(recover_config, 'advance')
        self.device = recover_config['device'] if 'device' in recover_config else 'cpu'
        self.int8_sequences = recover_config['int8_sequences']
        self.fp32_ops = recover_config['fp32_ops']
        self.bf16_ops = recover_config['bf16_ops']
        self.recipes = recover_config['recipes']
        self.quantized_node_info = []
        self._calibration_data = []
        self._fp32_print_data = []
        self.data_loader = data_loader
        self.recover_config = recover_config
        self._check_tf_version()
        self._check_args()
        self._gen_tmp_filenames()

        self._tmp_graph_def = copy.deepcopy(self.model.graph_def)
    # pylint: disable=no-member
    def _check_tf_version(self):
        is_supported_version = False
        try:
            from tensorflow import python
            if (hasattr(python, "pywrap_tensorflow")
                    and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):# pragma: no cover
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            elif hasattr(python.util, "_pywrap_util_port"):
                from tensorflow.python.util._pywrap_util_port import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (TF_SUPPORTED_MIN_VERSION <= tf.version.VERSION):
                is_supported_version = True

            if tf.version.VERSION == '2.6.0' and os.getenv('TF_ENABLE_ONEDNN_OPTS') == '1':
                is_supported_version = True
        except Exception as e:
            raise ValueError(e)
        finally:# pragma: no cover
            if tf.version.VERSION > TF_SUPPORTED_MAX_VERSION:
                logger.warning(
                    str('Please note the {} version of Intel® Optimizations for '
                        'TensorFlow is not fully verified! '
                        'Suggest to use the versions '
                        'between {} and {} if meet problem.').format(tf.version.VERSION,
                                                                     TF_SUPPORTED_MIN_VERSION,
                                                                     TF_SUPPORTED_MAX_VERSION))
            if tf.version.VERSION == '2.5.0' and os.getenv('TF_ENABLE_MKL_NATIVE_FORMAT') != '0':
                logger.warning("Please set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 "
                               "when Tensorflow 2.5.0 installed.")

            if tf.version.VERSION == '2.6.0' and os.getenv('TF_ENABLE_ONEDNN_OPTS') != '1':
                logger.warning("Please set environment variable TF_ENABLE_ONEDNN_OPTS=1 "
                               "when Tensorflow 2.6.0 installed.")

            if not is_supported_version:
                raise ValueError(
                    str('Please install Intel® Optimizations for TensorFlow '
                        'or MKL enabled source build TensorFlow '
                        'within version >={} and <={}.').format(TF_SUPPORTED_MIN_VERSION,
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
        self._tmp_model = Model(self.model._model, **self.model.kwargs)
        self._tmp_model.output_tensor_names = self.output_tensor_names
        self._tmp_model.input_tensor_names = self.input_tensor_names

    def convert_without_calib(self):
        model = self._tmp_model

        if len(self.op_wise_config) > 0:
            model = self.quantize_without_calib()

        if len(self.bf16_ops) > 0:
            model = self.bf16_convert()

        post_cse_graph_def = PostCseOptimizer(model.graph_def).do_transformation()
        post_cse_graph_def.library.CopyFrom(self.model.graph_def.library)
        model.graph_def = post_cse_graph_def

        if debug:
            model.save(self.output_graph)

        return model

    def _analysis_rnn_model(self):
        g = GraphAnalyzer()
        g.graph = self._tmp_graph_def
        graph_info = g.parse_graph()
        rnn_pattern = [['TensorArrayV3'], ['Enter'], ['TensorArrayReadV3'], \
            ['MatMul'], ['BiasAdd']]
        target_nodes = g.query_fusion_pattern_nodes(rnn_pattern)
        res = {}
        for i in target_nodes:
            if i[-3] not in self.bf16_ops and i[-3] not in self.fp32_ops:
                res[(i[-3], i[-2])] = graph_info[i[1]].node.attr['frame_name'].s.decode()

        return res

    def quantize_without_calib(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            self._quantize_graph()
            self._rnn_details = self._analysis_rnn_model()
            self._freeze_requantization_ranges_without_calib()
            self._fuse_requantize_with_fused_quantized_node()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._tmp_model = None
            logger.error('Fail to quantize graph due to {}.'.format(str(e)))
        finally:
            if not debug:
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
            logger.error('Fail to convert graph due to {}.'.format(str(e)))
        finally:
            if debug:
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

        self._tmp_graph_def, self.quantized_node_info = QuantizeGraphForIntel(
            self._tmp_graph_def,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device,
            False).do_transform()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        if debug:
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_dynamic_range_model_path)

    def _freeze_requantization_ranges_without_calib(self):

        self._tmp_graph_def = FreezeValueWithoutCalibTransformer(
            self._tmp_graph_def,
            self.recover_config,
            postfix='__min').do_transformation_without_calib()
        self._tmp_graph_def = FreezeValueWithoutCalibTransformer(
            self._tmp_graph_def,
            self.recover_config,
            postfix='__max').do_transformation_without_calib()
        self._tmp_graph_def = FreezeValueWithoutCalibTransformer(
            self._tmp_graph_def,
            self.recover_config,
            postfix='__requant_min_max',
            device = self.device).do_transformation_without_calib()

        self._tmp_graph_def = QuantizedRNNConverter(
            self._tmp_graph_def, self._calibration_data, self._rnn_details).do_transformation()

        if 'scale_propagation_max_pooling' in self.recipes and \
                self.recipes['scale_propagation_max_pooling']:
            self._tmp_graph_def = ScaleProPagationTransformer(
                self._tmp_graph_def).do_transformation()

        if debug:
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

        if 'scale_propagation_concat' in self.recipes and self.recipes['scale_propagation_concat']:
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

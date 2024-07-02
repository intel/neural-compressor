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
"""Graph Converter Class."""

import copy
import logging
import os
import tempfile
from collections import OrderedDict, UserDict

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.insert_print_node import InsertPrintMinMaxNode
from neural_compressor.model import Model
from neural_compressor.model.tensorflow_model import TensorflowSavedModelModel
from neural_compressor.utils.utility import (
    CaptureOutputToFile,
    CpuInfo,
    combine_histogram,
    deep_get,
    get_all_fp32_data,
    get_tensor_histogram,
)

from .graph_rewriter.bf16.bf16_convert import BF16Convert
from .graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from .graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer
from .graph_rewriter.generic.fuse_pad_with_fp32_conv import FusePadWithFP32Conv2DOptimizer
from .graph_rewriter.generic.remove_training_nodes import RemoveTrainingNodesOptimizer
from .graph_rewriter.generic.strip_equivalent_nodes import StripEquivalentNodesOptimizer
from .graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from .graph_rewriter.int8.freeze_fake_quant import FreezeFakeQuantOpOptimizer
from .graph_rewriter.int8.freeze_value import FreezeValueTransformer
from .graph_rewriter.int8.fuse_conv_redundant_dequantize import FuseConvRedundantDequantizeTransformer
from .graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_redundant_dequantize import FuseMatMulRedundantDequantizeTransformer
from .graph_rewriter.int8.fuse_matmul_requantize import (
    FuseMatMulRequantizeDequantizeNewAPITransformer,
    FuseMatMulRequantizeDequantizeTransformer,
    FuseMatMulRequantizeNewAPITransformer,
    FuseMatMulRequantizeTransformer,
)
from .graph_rewriter.int8.meta_op_optimizer import MetaInfoChangingMemOpOptimizer
from .graph_rewriter.int8.post_hostconst_converter import PostHostConstConverter
from .graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
from .graph_rewriter.int8.scale_propagation import ScaleProPagationTransformer
from .graph_rewriter.qdq.insert_qdq_pattern import GenerateGraphWithQDQPattern
from .graph_rewriter.qdq.merge_duplicated_qdq import MergeDuplicatedQDQOptimizer
from .graph_rewriter.qdq.share_qdq_y_pattern import ShareQDQForItexYPatternOptimizer
from .graph_util import GraphAnalyzer
from .graph_util import GraphRewriterHelper as Helper
from .quantize_graph.qdq.optimize_qdq import OptimizeQDQGraph
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph_common import QuantizeGraphHelper
from .transform_graph.bias_correction import BiasCorrection
from .transform_graph.insert_logging import InsertLogging
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .util import (
    TF_SPR_BASE_VERSIONS,
    generate_feed_dict,
    iterator_sess_run,
    version1_eq_version2,
    version1_gt_version2,
    version1_gte_version2,
    version1_lt_version2,
    version1_lte_version2,
)

TF_SUPPORTED_MAX_VERSION = "2.14.0"
TF_SUPPORTED_MIN_VERSION = "1.14.0"

logger = logging.getLogger("neural_compressor")
debug = bool(logger.level == logging.DEBUG)


class GraphConverter:
    """Graph Converter Class is used to generate the quantization graph."""

    def __init__(
        self,
        model,
        qt_config={},
        recipes={},
        int8_sequences={},
        fp32_ops=[],
        bf16_ops=[],
        data_loader=None,
        calib_func=None,
        fake_quant=False,
        itex_mode=False,
        qdq_enabled=False,
        new_api=False,
        performance_only=False,
        use_bf16=False,
    ):
        """Convert graph.

        :param model: input tensorflow model.
        :param qt_config: quantization configs, including iteration and op-wise quant config
        :param fp32_ops: fall back to fp32 dtype op list
        :param bf16_ops: fall back to bf16 dtype op list
        :param data_loader: for calibration phase used dataloader
        :param calib_func: for calibration phase used function
        :param fake_quant: for quantization-aware training model conversion to default model
        """
        self.model = model
        # (TODO) does it right to make the internal model format as graph_def
        self.output_tensor_names = self.model.output_tensor_names
        self.input_tensor_names = self.model.input_tensor_names
        # quantize specific config
        self.calib_iteration = qt_config["calib_iteration"] if not fake_quant else 0
        self.op_wise_config = qt_config["op_wise_config"]
        self.advance_config = deep_get(qt_config, "advance")
        self.device = qt_config["device"] if "device" in qt_config else "cpu"
        self.int8_sequences = int8_sequences
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops
        self.recipes = recipes
        self.fake_quant = fake_quant
        self.itex_mode = itex_mode
        self.qdq_enabled = qdq_enabled
        self.performance_only = performance_only
        self.quantized_node_info = []
        self._calibration_data = []
        self._fp32_print_data = []
        self.data_loader = data_loader
        self.calib_func = calib_func
        self._check_tf_version()
        self._check_args()

        if "backend" in self.model.kwargs:
            self._fp32_model = Model(self.model._model, **self.model.kwargs)
        else:
            self._fp32_model = Model(
                self.model._model,
                **self.model.kwargs,
                backend="itex" if itex_mode and not isinstance(self.model, TensorflowSavedModelModel) else "default"
            )
        self._fp32_model.graph_def = self.model.graph_def
        self._fp32_model.output_tensor_names = self.output_tensor_names
        self._fp32_model.input_tensor_names = self.input_tensor_names

        self._gen_tmp_filenames()
        self._kl_op_dict = {}
        self._kl_keys = []
        self._llm_weight_minmax = {}
        self._print_node_mapping = {}
        self._enable_kl_op_names = [k for k in self.op_wise_config if self.op_wise_config[k][1] == "kl"]
        self.scale_info = {}
        self.scale_info.update(qt_config)
        self.scale_info.update({"recipes": self.recipes})
        self.scale_info.update({"int8_sequences": self.int8_sequences})
        self.scale_info.update({"bf16_ops": self.bf16_ops})
        self.scale_info.update({"fp32_ops": self.fp32_ops})

        if "backend" in self.model.kwargs:
            self._sampling_model = Model(self.model._model, **self.model.kwargs)
        else:
            self._sampling_model = Model(
                self.model._model,
                **self.model.kwargs,
                backend="itex" if itex_mode and not isinstance(self.model, TensorflowSavedModelModel) else "default"
            )
        self._sampling_model.output_tensor_names = self.output_tensor_names
        self._sampling_model.input_tensor_names = self.input_tensor_names

        if self.performance_only:
            # reuse the fp32 model for performance only mode
            self._tmp_graph_def = self.model.graph_def
        else:
            self._tmp_graph_def = copy.deepcopy(self.model.graph_def)
        self.new_api = new_api  # bool(version1_gte_version2(tf.version.VERSION, '2.8.0'))
        self.use_bf16 = use_bf16
        self.exclude_node_names = []

    # pylint: disable=no-member
    def _inference(self, model):
        """Run the calibration on the input graph.

        Args:
            model(TensorflowBaseModel): input TensorflowBaseModel
        """
        if self.calib_func:
            self.calib_func(model.model)
            return

        if model.model_type == "llm_saved_model":
            self._inference_llm(model)
            return

        # ITEX optimization has broken INC calibration process.
        # INC needs turn off ITEX optimization pass in calibration stage.
        # TODO ITEX will provide API to replace setting environment variable.
        os.environ["ITEX_REMAPPER"] = "0"
        sess = model.sess
        iter_op = model.iter_op
        input_tensor = model.input_tensor
        output_tensor = model.output_tensor
        # TF table initialization: https://github.com/tensorflow/tensorflow/issues/8665
        node_names = [node.name for node in sess.graph.as_graph_def().node]
        if "init_all_tables" in node_names:
            init_table_op = sess.graph.get_operation_by_name("init_all_tables")
            sess.run(init_table_op)

        logger.info("Start sampling on calibration dataset.")
        if hasattr(self.data_loader, "__len__") and len(self.data_loader) == 0:
            feed_dict = {}
            _ = (
                sess.run(output_tensor, feed_dict)
                if iter_op == []
                else iterator_sess_run(sess, iter_op, feed_dict, output_tensor, self.calib_iteration)
            )
        for idx, (inputs, labels) in enumerate(self.data_loader):
            if len(input_tensor) == 1:
                feed_dict = {}
                if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name == t_name:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
                feed_dict = {}
                if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                    for name in inputs:
                        for tensor in input_tensor:
                            pos = tensor.name.rfind(":")
                            t_name = tensor.name if pos < 0 else tensor.name[:pos]
                            if name in [tensor.name, t_name]:
                                feed_dict[tensor] = inputs[name]
                                break
                else:
                    # sometimes the input_tensor is not the same order with inputs
                    # we should check and pair them
                    def check_shape(tensor, data):
                        # scalar or 1 dim default True
                        if (
                            tensor.shape is None
                            or tensor.shape == tf.TensorShape(None)
                            or len(tensor.shape.dims) == 1
                            or not hasattr(data, "shape")
                        ):
                            return True
                        tensor_shape = tuple(tensor.shape)
                        data_shape = tuple(data.shape)
                        for tensor_dim, data_dim in zip(tensor_shape, data_shape):
                            if tensor_dim is not None and tensor_dim != data_dim:
                                return False
                        return True

                    disorder_tensors = []
                    disorder_inputs = []
                    for idx, sort_tensor in enumerate(input_tensor):
                        sort_input = inputs[idx]
                        if check_shape(sort_tensor, sort_input):
                            feed_dict.update({sort_tensor: sort_input})
                        else:
                            disorder_tensors.append(sort_tensor)
                            disorder_inputs.append(sort_input)
                    for i, dis_tensor in enumerate(disorder_tensors):
                        for j, dis_input in enumerate(disorder_inputs):
                            if check_shape(dis_tensor, dis_input):
                                feed_dict.update({dis_tensor: dis_input})
                                break
            _ = (
                sess.run(output_tensor, feed_dict)
                if iter_op == []
                else iterator_sess_run(sess, iter_op, feed_dict, output_tensor, self.calib_iteration)
            )
            if idx + 1 == self.calib_iteration:
                break
        os.environ["ITEX_REMAPPER"] = "1"

    def _inference_llm(self, model):
        input_tensor_names = model.input_tensor_names
        auto_trackable = model.model
        infer = auto_trackable.signatures["serving_default"]
        for idx, (inputs, _) in enumerate(self.data_loader):
            feed_dict = {}
            if len(input_tensor_names) == 1:
                feed_dict[input_tensor_names[0]] = inputs
            else:
                assert len(input_tensor_names) == len(inputs), "inputs len must equal with input_tensor"
                for i, input_tensor_name in enumerate(input_tensor_names):
                    feed_dict[input_tensor_name] = inputs[i]

            _ = infer(**feed_dict)

            if idx >= self.calib_iteration:
                break

    def _check_tf_version(self):
        """Check if the installed tensorflow version is supported."""
        is_supported_version = False
        is_sprbase_version = False
        try:
            from tensorflow import python

            if hasattr(python, "pywrap_tensorflow") and hasattr(python.pywrap_tensorflow, "IsMklEnabled"):
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            elif hasattr(python.util, "_pywrap_util_port"):
                from tensorflow.python.util._pywrap_util_port import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (version1_lte_version2(TF_SUPPORTED_MIN_VERSION, tf.version.VERSION)):
                is_supported_version = True

            if version1_gte_version2(tf.version.VERSION, "2.6.0") and os.getenv("TF_ENABLE_ONEDNN_OPTS") == "1":
                is_supported_version = True

            if version1_gte_version2(tf.version.VERSION, "2.9.0"):
                is_supported_version = True

            if tf.version.VERSION == "1.15.0-up3":
                is_supported_version = True

            if tf.version.VERSION in TF_SPR_BASE_VERSIONS:
                is_supported_version = True
                is_sprbase_version = True

        except Exception as e:
            raise ValueError(e)
        finally:
            if version1_gt_version2(tf.version.VERSION, TF_SUPPORTED_MAX_VERSION) and not is_sprbase_version:
                logger.warning(
                    str(
                        "Please note the {} version of TensorFlow is not fully verified! "
                        "Suggest to use the versions between {} and {} if meet problem."
                    ).format(tf.version.VERSION, TF_SUPPORTED_MIN_VERSION, TF_SUPPORTED_MAX_VERSION)
                )

            if version1_eq_version2(tf.version.VERSION, "2.5.0") and os.getenv("TF_ENABLE_MKL_NATIVE_FORMAT") != "0":
                logger.fatal(
                    "Please set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 " "when TensorFlow 2.5.0 installed."
                )

            if (
                version1_gte_version2(tf.version.VERSION, "2.6.0")
                and version1_lt_version2(tf.version.VERSION, "2.9.0")
                and os.getenv("TF_ENABLE_ONEDNN_OPTS") != "1"
            ):
                logger.fatal(
                    "Please set environment variable TF_ENABLE_ONEDNN_OPTS=1 "
                    "when TensorFlow >= 2.6.0 and < 2.9.0 installed."
                )

            if not is_supported_version:
                raise ValueError(
                    str("Please install TensorFlow within version >={} and <={}.").format(
                        TF_SUPPORTED_MIN_VERSION, TF_SUPPORTED_MAX_VERSION
                    )
                )

    def _check_args(self):
        """Check model's arguments."""
        if (
            self.model.workspace_path
            and not os.path.isdir(self.model.workspace_path)
            and not os.path.exists(os.path.dirname(self.model.workspace_path))
        ):
            raise ValueError('"output_graph" directory does not exist.')
        self._output_path = self.model.workspace_path

    def _gen_tmp_filenames(self):
        """Generate the temporary file names."""
        self._int8_dynamic_range_model_path = os.path.join(self._output_path, "int8_dynamic_range_graph")
        self._int8_logged_model_path = os.path.join(self._output_path, "int8_logged_graph")
        self._fp32_logged_model_path = os.path.join(self._output_path, "fp32_logged_graph")
        self._int8_frozen_range_model_path = os.path.join(self._output_path, "int8_frozen_range_graph")
        self._bf16_mixed_precision_model_path = os.path.join(self._output_path, "int8_bf16_mixed_precision_graph")

        self.output_graph = os.path.join(self._output_path, "int8_final_fused_graph")
        if self.performance_only:
            # reuse the fp32 model for performance only mode
            self._tmp_model = self._fp32_model
        else:
            # to keep temp model
            if "backend" in self.model.kwargs:
                self._tmp_model = Model(self.model._model, **self.model.kwargs)
            else:
                self._tmp_model = Model(
                    self.model._model,
                    **self.model.kwargs,
                    backend=(
                        "itex"
                        if self.itex_mode and not isinstance(self.model, TensorflowSavedModelModel)
                        else "default"
                    )
                )
            self._tmp_model.graph_def = self.model.graph_def
            self._tmp_model.output_tensor_names = self.output_tensor_names
            self._tmp_model.input_tensor_names = self.input_tensor_names

    def convert(self):
        """Do conversion.

        Including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.
            5) bf16 convert if the self.bf16_ops is not empty

        :return:
        """
        model = self._tmp_model
        if len(self.op_wise_config) > 0:
            if self.qdq_enabled:
                model = self.quantize_with_qdq_pattern()
            else:
                model = self.quantize()

        if self.itex_mode:
            host_const_graph_def = PostHostConstConverter(self._tmp_model.graph_def).do_transformation()
            host_const_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._tmp_model.graph_def = host_const_graph_def

            return self._tmp_model

        if self.exclude_node_names:
            self.bf16_ops.extend(self.exclude_node_names)

        if (
            len(self.bf16_ops) > 0
            and (self.use_bf16 or self.performance_only)
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):
            model = self.bf16_convert()

        if self.new_api:
            if self.performance_only:
                model.graph_def = FuseConvRedundantDequantizeTransformer(model.graph_def).do_transformation()
            post_optimize_graph_def = FuseMatMulRedundantDequantizeTransformer(model.graph_def).do_transformation()
            post_optimize_graph_def.library.CopyFrom(self.model.graph_def.library)
            model.graph_def = post_optimize_graph_def
        post_cse_graph_def = PostCseOptimizer(model.graph_def).do_transformation()
        post_hostconst_graph_def = PostHostConstConverter(post_cse_graph_def).do_transformation()
        post_hostconst_graph_def.library.CopyFrom(self.model.graph_def.library)
        model.graph_def = post_hostconst_graph_def

        if debug:
            model.save(self.output_graph)
            logger.info("Save converted graph file to {}.".format(self.output_graph))
        model.q_config = self.scale_info
        return model

    def _get_fp32_print_node_names(self, specified_op_list):
        """Get the print node name of the fp32 graph."""
        offset_map = {
            "QuantizedConv2DWithBiasSumAndRelu": 3,
            "QuantizedConv2DWithBiasAndRelu": 2,
            "QuantizedConv2DWithBias": 1,
        }
        target_conv_op = []
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_model.graph_def, self._fp32_model.input_node_names, self._fp32_model.output_node_names
        )

        node_name_mapping = {node.name: node for node in self._tmp_graph_def.node if node.op != "Const"}

        for node in self._tmp_graph_def.node:
            if node.op in offset_map:
                target_conv_op.append(node.name.split("_eightbit_")[0])
        fp32_node_name_mapping = {node.name: node for node in sorted_graph.node if node.op != "Const"}
        sorted_node_names = [i.name for i in sorted_graph.node if i.op != "Const"]

        output_node_names = []
        for i in target_conv_op:
            if specified_op_list and i not in specified_op_list:
                continue
            if node_name_mapping[i + "_eightbit_quantized_conv"].op == "QuantizedConv2DWithBiasSumAndRelu":
                start_index = sorted_node_names.index(i)
                for index, value in enumerate(sorted_node_names[start_index:]):
                    if (
                        fp32_node_name_mapping[value].op.startswith("Add")
                        and fp32_node_name_mapping[sorted_node_names[start_index + index + 1]].op == "Relu"
                    ):
                        output_node_names.append(sorted_node_names[start_index + index + 1])
                        self._print_node_mapping[sorted_node_names[start_index + index + 1]] = i

            elif i in sorted_node_names:
                start_index = sorted_node_names.index(i)
                end_index = start_index + offset_map[node_name_mapping[i + "_eightbit_quantized_conv"].op]
                output_node_names.append(sorted_node_names[end_index])
                self._print_node_mapping[sorted_node_names[end_index]] = i

        for i in output_node_names:
            self._kl_keys.append(";" + i + "__print__;__KL")

        fp32_graph_def = graph_pb2.GraphDef()
        fp32_graph_def.CopyFrom(self._fp32_model.graph_def)
        self._fp32_model.graph_def = InsertLogging(
            self._fp32_model.graph_def, node_name_list=output_node_names, message="__KL:", summarize=-1, dump_fp32=True
        ).do_transformation()

        self._fp32_model.save(self._fp32_logged_model_path)
        self._fp32_model.graph_def = fp32_graph_def
        return self._fp32_model

    def _search_y_pattern_for_itex(self):
        """Search the Y pattern for itex and return the op name."""
        g = GraphAnalyzer()
        g.graph = self._fp32_model.graph_def
        g.parse_graph()
        y_pattern = [["Conv2D", "MatMul"], ["BiasAdd"], ["Add", "AddV2", "AddN"], ("Relu",)]
        y_pattern_variant = [["MaxPool", "AvgPool"], ["Add", "AddV2", "AddN"], ("Relu",)]
        target_nodes = g.query_fusion_pattern_nodes(y_pattern)
        target_nodes_variant = g.query_fusion_pattern_nodes(y_pattern_variant)

        res = {}
        for i in target_nodes:
            if i[2] not in res:
                res[i[2]] = 1
            else:
                res[i[2]] += 1
        matched_add_nodes = [(i,) for i in res if res[i] == 2]
        for i in res:
            if res[i] == 1:
                for j in target_nodes_variant:
                    if j[1] == i:
                        matched_add_nodes.append((i,))
        return matched_add_nodes

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph).

        Including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            self._quantize_graph()
            self.quantized_node_info = [tuple(i) for i in self.quantized_node_info]

            if self.fake_quant:  # pragma: no cover
                self._fuse_requantize_with_fused_quantized_node()
            else:
                if self._enable_kl_op_names:
                    self._get_fp32_print_node_names(self._enable_kl_op_names)
                    self._generate_calibration_data(self._fp32_logged_model_path, self._fp32_print_data, True)

                output_tensor_names = copy.deepcopy(self.model.output_tensor_names)
                sampling_graph_def = copy.deepcopy(self._fp32_model.graph_def)

                # TODO: this is a workaround to make Min/Max node be completely eliminated in int8 graph
                # after enabling pad+conv2d in new API.
                non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))
                sampling_graph_def = FusePadWithFP32Conv2DOptimizer(
                    sampling_graph_def, non_pad_ops, self._tmp_model.input_node_names, self.op_wise_config, self.new_api
                ).do_transformation()

                for i in self.quantized_node_info:
                    sampling_graph_def, output_names = InsertPrintMinMaxNode(
                        sampling_graph_def, i[0], i[-1], self.new_api
                    ).do_transformation()
                    output_tensor_names.extend(output_names)
                if self.quantized_node_info:
                    sampling_graph_def.library.CopyFrom(self.model.graph_def.library)
                    self._sampling_model.graph_def = sampling_graph_def
                    self._sampling_model.output_tensor_names = output_tensor_names
                    tmp_dump_file = tempfile.mkstemp(suffix=".log")[1]
                    with CaptureOutputToFile(tmp_dump_file):
                        self._inference(self._sampling_model)
                    self._calibration_data = Helper.gen_valid_sampling_log(tmp_dump_file)

                del output_tensor_names
                del sampling_graph_def
                del self._sampling_model
                import gc

                gc.collect()

                if len(self._calibration_data) > 0:
                    self._freeze_requantization_ranges(self._kl_op_dict)
                    self._fuse_requantize_with_fused_quantized_node()

        except ValueError as e:
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
            self._tmp_model = None
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            self._tmp_model = None
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
            return self._tmp_model
        finally:
            if not debug:
                self._post_clean()
        return self._tmp_model

    def bf16_convert(self):
        """Convert fp32 nodes in bf16_node to bf16 dtype based on FP32 + INT8 mixed precision graph."""
        try:
            logger.info("Start BF16 conversion.")
            self._tmp_model.graph_def = BF16Convert(
                self._tmp_model.graph_def, self.fp32_ops, self.bf16_ops
            ).do_transformation()

        except Exception as e:
            import traceback

            traceback.print_exc()
            self._tmp_model = None
            logger.error("Fail to convert graph due to {}.".format(str(e)))
        finally:
            if debug:
                self._tmp_model.save(self._bf16_mixed_precision_model_path)

            return self._tmp_model

    def _quantize_graph(self):
        """Quantize graph."""
        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))

        self._tmp_graph_def = FusePadWithConv2DOptimizer(
            self._tmp_graph_def, non_pad_ops, self._tmp_model.input_node_names, self.op_wise_config, self.new_api
        ).do_transformation()

        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(
            self._tmp_graph_def, self._tmp_model.input_node_names, self._tmp_model.output_node_names
        )

        self._tmp_graph_def, self.quantized_node_info, exclude_node_names = QuantizeGraphForIntel(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device,
            self.fake_quant,
            self.new_api,
            self.performance_only,
            self.itex_mode,
        ).do_transform()
        self.exclude_node_names = exclude_node_names
        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
        if debug and not self.performance_only:
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_dynamic_range_model_path)

    def _generate_calibration_data(self, tmp_path, output_data, enable_kl_algo=False):
        """Generate the calibration data."""
        tmp_dump_file = os.path.join(os.path.dirname(self.output_graph), "requant_min_max.log")

        logger.debug("Generate calibration data and save to {}.".format(tmp_dump_file))

        if "backend" in self._tmp_model.kwargs:
            model = Model(tmp_path, **self._tmp_model.kwargs)
        else:
            model = Model(
                tmp_path,
                **self._tmp_model.kwargs,
                backend=(
                    "itex"
                    if self.itex_mode and not isinstance(self._tmp_model, TensorflowSavedModelModel)
                    else "default"
                )
            )
        model.output_tensor_names = self.output_tensor_names
        model.input_tensor_names = self.input_tensor_names

        with CaptureOutputToFile(tmp_dump_file):
            self._inference(model)

        with open(tmp_dump_file, errors="ignore") as f:
            output_data.extend(f.readlines())

        for line in output_data:
            if enable_kl_algo and line.rsplit(":")[0] in self._kl_keys:
                fp32_data = get_all_fp32_data(line.rsplit(":")[-1])
                key = self._print_node_mapping[line[1:].split("__print")[0]] + "_eightbit_requant_range"
                if key not in self._kl_op_dict:
                    self._kl_op_dict[key] = get_tensor_histogram(fp32_data)
                else:
                    self._kl_op_dict[key] = combine_histogram(self._kl_op_dict[key], fp32_data)

    def _freeze_requantization_ranges(self, additional_data=None):
        """Freeze requantization ranges after doing quantization."""
        self._tmp_graph_def, quantizev2_max = FreezeValueTransformer(
            self._tmp_graph_def, self._calibration_data, "__max:", device=self.device
        ).do_transformation()
        self._tmp_graph_def, quantizev2_min = FreezeValueTransformer(
            self._tmp_graph_def, self._calibration_data, "__min:", device=self.device
        ).do_transformation()
        self._tmp_graph_def, requant_min_max = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            "__requant_min_max",
            tensor_data=additional_data,
            device=self.device,
        ).do_transformation()

        self.scale_info.update(quantizev2_max)
        self.scale_info.update(quantizev2_min)
        self.scale_info.update(requant_min_max)

        if "scale_propagation_max_pooling" in self.recipes and self.recipes["scale_propagation_max_pooling"]:
            self._tmp_graph_def = ScaleProPagationTransformer(self._tmp_graph_def).do_transformation()

        if debug and not self.new_api:
            self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.save(self._int8_frozen_range_model_path)

    def _fuse_requantize_with_fused_quantized_node(self):
        """Fuse the Requantize/Dequantize with fused quantized Ops."""
        if self.fake_quant:  # pragma: no cover
            self._tmp_graph_def = FreezeFakeQuantOpOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def, self.device, self.new_api
        ).do_transformation()

        if not self.fake_quant:
            if self.qdq_enabled:
                self._tmp_graph_def = FuseMatMulRequantizeNewAPITransformer(self._tmp_graph_def).do_transformation()

                self._tmp_graph_def = FuseMatMulRequantizeDequantizeNewAPITransformer(
                    self._tmp_graph_def
                ).do_transformation()
            else:
                self._tmp_graph_def = FuseMatMulRequantizeTransformer(self._tmp_graph_def).do_transformation()

                self._tmp_graph_def = FuseMatMulRequantizeDequantizeTransformer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def, self._tmp_model.input_node_names, self._tmp_model.output_node_names
        ).do_transformation()

        input_output_names = self._tmp_model.input_node_names + self._tmp_model.output_node_names
        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def, protected_nodes=input_output_names
        ).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()

        if self.performance_only or (
            "scale_propagation_concat" in self.recipes and self.recipes["scale_propagation_concat"]
        ):
            self._tmp_graph_def = RerangeQuantizedConcat(
                self._tmp_graph_def, self.device, performance_only=self.performance_only
            ).do_transformation()

        self._tmp_graph_def = MetaInfoChangingMemOpOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripEquivalentNodesOptimizer(
            self._tmp_graph_def, self._tmp_model.output_node_names
        ).do_transformation()

        if self.advance_config is not None and deep_get(self.advance_config, "bias_correction") is not None:
            self._tmp_graph_def = BiasCorrection(
                self._tmp_graph_def, self.model.graph_def, self.new_api
            ).do_transformation()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        self._tmp_model.graph_def = self._tmp_graph_def

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if os.path.exists(self._int8_logged_model_path) and os.path.isdir(self._int8_logged_model_path):
            import shutil

            shutil.rmtree(self._int8_logged_model_path)

        elif gfile.Exists(self._int8_logged_model_path + ".pb"):
            os.remove(self._int8_logged_model_path + ".pb")

    def quantize_with_qdq_pattern(self):
        """Quantize model by inserting QDQ.

        step 1: insert QDQ pairs and update node info
        step 2: convert Q-DQ-node-Q-DQ to Q-newAPI node-DQ
        """
        try:
            self._insert_qdq_pairs()
            self._convert_qdq()

        except ValueError as e:
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
            self._tmp_model = None
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            self._tmp_model = None
            logger.error("Fail to quantize graph due to {}.".format(str(e)))
            return self._tmp_model
        finally:
            if not debug:
                self._post_clean()
        return self._tmp_model

    def _insert_qdq_pairs(self):
        """Insert QDQ pairs before Conv/MatMul/Pooling Ops."""
        # Fuse Pad into Conv2D, Conv3D, DepthwiseConv2dNative
        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))
        self._tmp_graph_def = FusePadWithConv2DOptimizer(
            self._tmp_graph_def, non_pad_ops, self._tmp_model.input_node_names, self.op_wise_config, self.new_api, True
        ).do_transformation()

        # Sort graph
        self._tmp_graph_def = QuantizeGraphHelper().get_sorted_graph(
            self._tmp_graph_def, self._tmp_model.input_node_names, self._tmp_model.output_node_names
        )

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        # Find out the quantized nodes
        self.quantized_node_info = OptimizeQDQGraph(
            self._tmp_graph_def,
            self._tmp_model.input_node_names,
            self._tmp_model.output_node_names,
            self.op_wise_config,
            self.int8_sequences,
            self.device,
            self.fake_quant,
            self.new_api,
            self.performance_only,
            self.itex_mode,
        ).get_quantized_nodes()

        if self.itex_mode:
            self.quantized_node_info.extend(self._search_y_pattern_for_itex())

        if self._enable_kl_op_names:
            self._get_fp32_print_node_names(self._enable_kl_op_names)
            self._generate_calibration_data(self._fp32_logged_model_path, self._fp32_print_data, True)

        # Calibration using sampling model
        output_tensor_names = copy.deepcopy(self.model.output_tensor_names)
        sampling_graph_def = copy.deepcopy(self._fp32_model.graph_def)
        # TODO: this is a workaround to make Min/Max node be completely eliminated in int8 graph
        # after enabling pad+conv2d in new API.
        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))
        sampling_graph_def = FusePadWithFP32Conv2DOptimizer(
            sampling_graph_def, non_pad_ops, self._tmp_model.input_node_names, self.op_wise_config, self.new_api, True
        ).do_transformation()

        for i in self.quantized_node_info:
            sampling_graph_def, output_names = InsertPrintMinMaxNode(
                sampling_graph_def, i[0], i[-1], self.new_api
            ).do_transformation()
            output_tensor_names.extend(output_names)

        if self.quantized_node_info:
            sampling_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._sampling_model.graph_def = sampling_graph_def
            self._sampling_model.output_tensor_names = output_tensor_names
            tmp_dump_file = tempfile.mkstemp(suffix=".log")[1]
            with CaptureOutputToFile(tmp_dump_file):
                self._inference(self._sampling_model)
            self._calibration_data = Helper.gen_valid_sampling_log(tmp_dump_file)

        if hasattr(self._sampling_model, "_weight_tensor_minmax_dict"):
            self._llm_weight_minmax = self._sampling_model.weight_tensor_minmax_dict

        del sampling_graph_def
        del output_tensor_names
        del self._sampling_model
        import gc

        gc.collect()

        # Insert QDQ pattern
        self._tmp_graph_def = GenerateGraphWithQDQPattern(
            self._tmp_graph_def,
            self._calibration_data,
            self.op_wise_config,
            self.fake_quant,
            self.fp32_ops,
            self.bf16_ops,
            self.quantized_node_info,
            self.device,
            self.performance_only,
            self.itex_mode,
            self._llm_weight_minmax,
        ).do_transformation()

    def _convert_qdq(self):
        """Convert Dequantize + Op + QuantizeV2 into QuantizedOps."""
        if self.itex_mode:
            self._tmp_graph_def, quantizev2_max = FreezeValueTransformer(
                self._tmp_graph_def, self._calibration_data, "__max:", self.itex_mode
            ).do_transformation()
            self._tmp_graph_def, quantizev2_min = FreezeValueTransformer(
                self._tmp_graph_def, self._calibration_data, "__min:", self.itex_mode
            ).do_transformation()
            self._tmp_graph_def, requant_min_max = FreezeValueTransformer(
                self._tmp_graph_def,
                self._calibration_data,
                "__requant_min_max",
                tensor_data=self._kl_op_dict,
                device=self.device,
                itex_mode=self.itex_mode,
            ).do_transformation()

            self.scale_info.update(quantizev2_max)
            self.scale_info.update(quantizev2_min)
            self.scale_info.update(requant_min_max)

            self._tmp_graph_def = StripUnusedNodesOptimizer(
                self._tmp_graph_def, self._tmp_model.input_node_names, self._tmp_model.output_node_names
            ).do_transformation()

            self._tmp_graph_def = ShareQDQForItexYPatternOptimizer(self._tmp_graph_def).do_transformation()
            self._tmp_graph_def = MergeDuplicatedQDQOptimizer(self._tmp_graph_def).do_transformation()

            self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)
            self._tmp_model.graph_def = self._tmp_graph_def
            self._tmp_model.graph_def.library.CopyFrom(self.model.graph_def.library)
        else:
            self._tmp_graph_def, exclude_node_names = OptimizeQDQGraph(
                self._tmp_graph_def,
                self._tmp_model.input_node_names,
                self._tmp_model.output_node_names,
                self.op_wise_config,
                self.int8_sequences,
                self.device,
                self.fake_quant,
                self.new_api,
                self.performance_only,
                self.itex_mode,
            ).do_transform()
            self.exclude_node_names = exclude_node_names

            if len(self._calibration_data) > 0:
                self._freeze_requantization_ranges(self._kl_op_dict)
                self._fuse_requantize_with_fused_quantized_node()

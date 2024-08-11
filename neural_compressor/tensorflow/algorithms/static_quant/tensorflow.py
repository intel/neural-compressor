#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""Tensorflow Adaptor Classes."""

import copy
import math
import os
import re
from collections import OrderedDict, UserDict
from copy import deepcopy
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
import yaml
from pkg_resources import parse_version

from neural_compressor.common import logger
from neural_compressor.tensorflow.quantization.config import StaticQuantConfig
from neural_compressor.tensorflow.utils import (
    SPR_BASE_VERSIONS,
    UNIFY_OP_TYPE_MAPPING,
    BaseDataLoader,
    BaseModel,
    CpuInfo,
    Statistics,
    deep_get,
    dump_elapsed_time,
    singleton,
    version1_eq_version2,
    version1_gte_version2,
    version1_lt_version2,
)

spr_base_verions = SPR_BASE_VERSIONS


class TensorFlowAdaptor:
    """Adaptor Layer for stock tensorflow and spr-base."""

    unify_op_type_mapping = UNIFY_OP_TYPE_MAPPING

    def __init__(self, framework_specific_info):
        """Initialization.

        Args:
            framework_specific_info: framework specific info passed from strategy.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.quantize_config = {"op_wise_config": {}}
        self.framework_specific_info = framework_specific_info
        self.approach = deep_get(self.framework_specific_info, "approach", False)
        self.device = self.framework_specific_info["device"]
        self.work_dir = os.path.abspath(self.framework_specific_info["workspace_path"])
        self.recipes = deep_get(self.framework_specific_info, "recipes", {})
        self.performance_only = deep_get(self.framework_specific_info, "performance_only", False)
        self.use_bf16 = deep_get(self.framework_specific_info, "use_bf16", False)
        self.backend = self.framework_specific_info["backend"]
        self.format = self.framework_specific_info["format"]
        os.makedirs(self.work_dir, exist_ok=True)

        self.model = None
        self.pre_optimized_model = None
        self.pre_optimizer_handle = None

        self.bf16_ops = []
        self.fp32_ops = []
        self.smooth_quant_mul_ops = []
        self.dump_times = 0  # for tensorboard

        cfg_yaml_name = "{}.yaml".format(self.__class__.__name__[: -len("Adaptor")].lower())
        self.itex_mode = self.backend == "itex" or cfg_yaml_name == "tensorflow_itex.yaml"

        if self.itex_mode:  # pragma: no cover
            self._check_itex()

        self.query_handler = TensorflowQuery(
            local_config_file=os.path.join(os.path.dirname(__file__), cfg_yaml_name),
            performance_only=self.performance_only,
            itex_mode=self.itex_mode,
        )

        self.new_api = tf.version.VERSION in spr_base_verions
        self.qdq_enabled = self.itex_mode or self.format == "QDQ" or self.new_api
        self.op_wise_sequences = self.query_handler.get_eightbit_patterns(self.qdq_enabled)

        self.fp32_results = []
        self.fp32_preds_as_label = False
        # self.benchmark = GLOBAL_STATE.STATE == MODE.BENCHMARK
        self.benchmark = False
        self.callbacks = []

        self.optype_statistics = None

        self._last_dequantize_ops = None

    def _check_itex(self):  # pragma: no cover
        try:
            import intel_extension_for_tensorflow
        except:
            raise ImportError(
                "The Intel® Extension for TensorFlow is not installed. "
                "Please install it to run models on ITEX backend"
            )

    def _tuning_cfg_to_fw(self, tuning_cfg):
        """Parse the neural_compressor wrapped configuration to Tensorflow.

        Args:
            tuning_cfg (dict): configuration for quantization.
        """
        self.quantize_config["calib_iteration"] = tuning_cfg["calib_iteration"]
        self.quantize_config["device"] = self.device
        self.quantize_config["advance"] = deep_get(tuning_cfg, "advance")
        fp32_ops = []
        bf16_ops = []
        dispatched_op_names = [j[0] for j in tuning_cfg["op"]]

        invalid_op_names = [i for i in self.quantize_config["op_wise_config"] if i not in dispatched_op_names]

        for op_name in invalid_op_names:  # pragma: no cover
            self.quantize_config["op_wise_config"].pop(op_name)

        for each_op_info in tuning_cfg["op"]:
            op_name = each_op_info[0]

            if tuning_cfg["op"][each_op_info]["activation"]["dtype"] in ["fp32", "bf16"]:
                if op_name in self.quantize_config["op_wise_config"]:
                    self.quantize_config["op_wise_config"].pop(op_name)
                if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "fp32":
                    fp32_ops.append(op_name)
                if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "bf16":  # pragma: no cover
                    bf16_ops.append(op_name)
                continue

            is_perchannel = False
            bit = None
            if "weight" in tuning_cfg["op"][each_op_info]:
                is_perchannel = tuning_cfg["op"][each_op_info]["weight"]["granularity"] == "per_channel"
                # bit = tuning_cfg['op'][each_op_info]['weight']['bit']
            weight_bit = bit if bit else 7.0

            algorithm = tuning_cfg["op"][each_op_info]["activation"]["algorithm"]

            is_asymmetric = False
            if "activation" in tuning_cfg["op"][each_op_info]:
                is_asymmetric = tuning_cfg["op"][each_op_info]["activation"]["scheme"] == "asym"
            self.quantize_config["op_wise_config"][op_name] = (is_perchannel, algorithm, is_asymmetric, weight_bit)

        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops

    @dump_elapsed_time("Pass quantize model")
    def quantize(
        self,
        quant_config: StaticQuantConfig,
        model: BaseModel,
        calib_dataloader: Callable = None,
        calib_iteration: int = 100,
        calib_func: Callable = None,
    ):
        """Execute the quantize process on the specified model.

        Args:
            quant_config: a quantization configuration.
            model: the fp32 model to be quantized.
            calib_dataloader: a data loader for calibration.
            calib_iteration: the iteration of calibration.
            calib_func: the function used for calibration, should be a substitution for calib_dataloader
            when the built-in calibration function of INC does not work for model inference.

        Returns:
            converted_model: the quantized INC model wrapper.
        """
        assert (
            self.approach != "post_training_dynamic_quant"
        ), "Dynamic Quantization is not supported on TensorFlow framework now!"

        assert (
            self.approach != "quant_aware_training"
        ), "Quantize Aware Training is not supported on TensorFlow framework now!"

        self.calib_sampling_size = calib_dataloader.batch_size * calib_iteration if calib_dataloader else 100
        tune_cfg = self.parse_quant_config(quant_config, model, calib_iteration)
        self._tuning_cfg_to_fw(tune_cfg)
        self.bf16_ops.extend(self.smooth_quant_mul_ops)
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        from neural_compressor.tensorflow.quantization.utils.graph_converter import GraphConverter

        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
        if isinstance(calib_dataloader, BaseDataLoader):
            batch_size = calib_dataloader.batch_size
            try:
                for i in range(batch_size):
                    if calib_sampling_size % (batch_size - i) == 0:
                        calib_batch_size = batch_size - i
                        if i != 0:  # pragma: no cover
                            logger.warning(
                                "Reset `calibration.dataloader.batch_size` field "
                                "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                "divisible exactly by batch size"
                            )
                        break
                tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
                calib_dataloader.batch(calib_batch_size)
                self.quantize_config["calib_iteration"] = tmp_iterations
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=calib_dataloader,
                    calib_func=calib_func,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
            except Exception:  # pragma: no cover
                from neural_compressor.tensorflow.quantization.utils.utility import get_model_input_shape

                batch_size = get_model_input_shape(model)
                logger.warning(
                    "Fail to forward with batch size={}, set to {} now.".format(calib_dataloader.batch_size, batch_size)
                )
                calib_dataloader.batch(batch_size)
                self.quantize_config["calib_iteration"] = calib_sampling_size
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=calib_dataloader,
                    calib_func=calib_func,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
        else:  # pragma: no cover
            if hasattr(calib_dataloader, "batch_size") and calib_sampling_size % calib_dataloader.batch_size != 0:
                iter = self.quantize_config["calib_iteration"]
                logger.warning(
                    "Please note that calibration sampling size {} "
                    "isn't divisible exactly by batch size {}. "
                    "So the real sampling size is {}.".format(
                        calib_sampling_size, calib_dataloader.batch_size, calib_dataloader.batch_size * iter
                    )
                )
            converted_model = GraphConverter(
                model,
                qt_config=self.quantize_config,
                recipes=self.recipes,
                int8_sequences=self.op_wise_sequences,
                fp32_ops=self.fp32_ops,
                bf16_ops=self.bf16_ops,
                data_loader=calib_dataloader,
                calib_func=calib_func,
                qdq_enabled=self.qdq_enabled,
                new_api=self.new_api,
                performance_only=self.performance_only,
                use_bf16=self.use_bf16,
            ).convert()
        # just save framework_specific_info feature for recover
        converted_model.q_config.update({"framework_specific_info": self.framework_specific_info})

        self._dump_model_op_stats(converted_model.graph_def)

        return converted_model

    def _dump_model_op_stats(self, model_graphdef):
        """Dump the whole model's OPs statistics information for analysis."""
        fp32_op_list_uint8 = copy.deepcopy(self.query_handler.get_op_types_by_precision(precision="uint8"))
        fp32_op_list_int8 = copy.deepcopy(self.query_handler.get_op_types_by_precision(precision="int8"))
        fp32_op_list = list(set(fp32_op_list_uint8).union(set(fp32_op_list_int8)))

        int8_op_prefix_list = [
            "QuantizedConv2D",
            "_FusedQuantizedConv3D",
            "QuantizedDepthwise",
            "QuantizedMaxPool",
            "QuantizedAvgPool",
            "QuantizedConcatV2",
            "QuantizedMatMul",
            "_QuantizedFusedBatchNorm",
            "_QuantizedMatMul",
            "_QuantizedBatchMatMul",
            "_QuantizedFusedInstanceNorm",
            "_FusedQuantizedDeconv2D",
            "_FusedQuantizedDeconv3D",
        ]
        from tensorflow.python.framework import dtypes

        res = {}
        for op_type in fp32_op_list:
            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["QuantizeV2"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["Dequantize"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["Cast"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        fp32_op_list.extend(["QuantizeV2", "Dequantize", "Cast"])
        for i in model_graphdef.node:
            if i.op == "Const":
                continue
            possible_int8_res = [name for name in int8_op_prefix_list if i.op.find(name) != -1]

            if any(possible_int8_res):  # pragma: no cover
                origin_op_type = possible_int8_res[0].split("Quantized")[-1]
                if origin_op_type == "FusedBatchNorm":
                    origin_op_type = "FusedBatchNormV3"
                if origin_op_type == "FusedInstanceNorm":
                    origin_op_type = "_MklFusedInstanceNorm"
                if origin_op_type == "Depthwise":
                    origin_op_type = "DepthwiseConv2dNative"
                if origin_op_type == "BatchMatMul":
                    origin_op_type = "BatchMatMulV2"
                if origin_op_type == "FusedBatchMatMulV2":
                    origin_op_type = "_MklFusedBatchMatMulV2"
                if origin_op_type == "Deconv2D":
                    origin_op_type = "Conv2DBackpropInput"
                if origin_op_type == "Deconv3D":
                    origin_op_type = "Conv3DBackpropInputV2"
                res[origin_op_type]["INT8"] += 1

            if i.op in fp32_op_list:
                if "T" not in i.attr and i.op != "Cast":  # pragma: no cover
                    continue
                if i.op == "Cast":
                    if i.attr["DstT"].type == dtypes.bfloat16:
                        res[i.op]["BF16"] += 1
                    elif i.attr["DstT"].type == dtypes.float32:
                        res[i.op]["FP32"] += 1
                elif i.attr["T"].type == dtypes.bfloat16:
                    res[i.op]["BF16"] += 1
                elif i.attr["T"].type in (dtypes.quint8, dtypes.qint8):
                    res[i.op]["INT8"] += 1
                else:
                    res[i.op]["FP32"] += 1

        field_names = ["Op Type", "Total", "INT8", "BF16", "FP32"]
        output_data = [
            [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
            for op_type in fp32_op_list
        ]

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _query_bf16_ops(self, matched_nodes):
        """Collect the bf16 OPs configuration for quantization."""
        self.bf16_op_details = OrderedDict()

        valid_precision = self.query_handler.get_mixed_precision_combination()
        if ("bf16" in valid_precision and CpuInfo().bf16) or os.getenv("FORCE_BF16") == "1":
            for details in matched_nodes:
                node_op = details[-1][0]
                node_name = details[0]

                self.bf16_op_details[(node_name, node_op)] = [
                    {"weight": {"dtype": ["bf16"]}, "activation": {"dtype": ["bf16"]}},
                    {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}},
                ]

    def _query_quantizable_ops(self, matched_nodes):
        """Collect the op-wise configuration for quantization.

        Returns:
            OrderDict: op-wise configuration.
        """
        bf16_common_config = {"weight": {"dtype": "bf16"}, "activation": {"dtype": "bf16"}}
        fp32_common_config = {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}}
        uint8_type = self.query_handler.get_op_types_by_precision(precision="uint8")
        int8_type = self.query_handler.get_op_types_by_precision(precision="int8")
        bf16_type = self.query_handler.get_op_types_by_precision(precision="bf16")
        tf_quantizable_op_type = list(set(uint8_type).union(set(int8_type)))

        valid_precision = self.query_handler.get_mixed_precision_combination()
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability["Conv2D"])
        conv3d_config = copy.deepcopy(op_capability["Conv3D"]) if "Conv3D" in op_capability else None
        matmul_config = copy.deepcopy(op_capability["MatMul"])
        other_config = copy.deepcopy(op_capability["default"])

        self.quantizable_op_details = OrderedDict()
        self.recipes_ops = {}

        self._init_op_stat = {i: [] for i in tf_quantizable_op_type}

        exclude_first_quantizable_op = (
            True
            if "first_conv_or_matmul_quantization" in self.recipes
            and not self.recipes["first_conv_or_matmul_quantization"]
            else False
        )
        for details in matched_nodes:
            node_op = details[-1][0]
            node_name = details[0]
            patterns = details[-1]
            pat_length = len(patterns)
            pattern_info = {
                "sequence": [[",".join(patterns[: pat_length - i]) for i in range(pat_length)][0]],
                "precision": ["int8"],
            }
            first_conv_or_matmul_node = []
            if (
                node_op in tf_quantizable_op_type
                and node_name not in self.exclude_node_names
                and (node_name, self.unify_op_type_mapping[node_op]) not in self.quantizable_op_details
            ):
                if (
                    self.unify_op_type_mapping[node_op].find("conv2d") != -1
                    or self.unify_op_type_mapping[node_op].find("matmul") != -1
                ) and len(first_conv_or_matmul_node) == 0:
                    first_conv_or_matmul_node.append((node_name, self.unify_op_type_mapping[node_op]))
                    self.recipes_ops["first_conv_or_matmul_quantization"] = first_conv_or_matmul_node
                if exclude_first_quantizable_op and (  # pragma: no cover
                    self.unify_op_type_mapping[node_op].find("conv2d") != -1
                    or self.unify_op_type_mapping[node_op].find("matmul") != -1
                ):
                    exclude_first_quantizable_op = False
                    self.exclude_node_names.append(node_name)
                    continue
                self._init_op_stat[node_op].append(node_name)
                if self.unify_op_type_mapping[node_op].find("conv2d") != -1:
                    conv2d_int8_config = copy.deepcopy(conv_config)
                    conv2d_int8_config["pattern"] = pattern_info
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        conv2d_int8_config,
                        fp32_common_config,
                    ]
                elif self.unify_op_type_mapping[node_op].find("conv3d") != -1:
                    conv3d_int8_config = copy.deepcopy(conv3d_config)
                    conv3d_int8_config["pattern"] = pattern_info
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        conv3d_int8_config,
                        fp32_common_config,
                    ]
                elif self.unify_op_type_mapping[node_op].find("matmul") != -1:
                    matmul_int8_config = copy.deepcopy(matmul_config)
                    matmul_int8_config["pattern"] = pattern_info
                    # TODO enable the sym mode once the tf fixed the mkldequantize_op.cc bug.
                    # is_positive_input = self.pre_optimizer_handle.has_positive_input(node_name)
                    # matmul_scheme = 'sym' if is_positive_input else 'asym'
                    matmul_scheme = ["asym"]
                    matmul_int8_config["activation"]["scheme"] = matmul_scheme
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        matmul_int8_config,
                        fp32_common_config,
                    ]
                else:
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        copy.deepcopy(other_config),
                        fp32_common_config,
                    ]
                if node_op in bf16_type and (
                    ("bf16" in valid_precision and CpuInfo().bf16) or os.getenv("FORCE_BF16") == "1"
                ):
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])].insert(
                        1, bf16_common_config
                    )

                self.quantize_config["op_wise_config"][node_name] = (False, "minmax", False)
        return self.quantizable_op_details

    def _filter_unquantizable_concat(self, matched_nodes):
        """Filter out unquantizable ConcatV2 Ops based on the positive input rule."""
        target_concat_nodes = [i[0] for i in matched_nodes if i[-1][0] == "ConcatV2"]
        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper
        from neural_compressor.tensorflow.quantization.utils.utility import GraphAnalyzer

        g = GraphAnalyzer()
        g.graph = self.pre_optimized_model.graph_def
        graph_info = g.parse_graph()
        concat_nodes = g.query_fusion_pattern_nodes([["ConcatV2"]])
        for i in concat_nodes:
            concat_node_name = i[0]
            if concat_node_name not in target_concat_nodes:  # pragma: no cover
                continue
            input_positive_status = []
            for index in range(graph_info[concat_node_name].node.attr["N"].i):
                each_input_name = GraphRewriterHelper.node_name_from_input(
                    graph_info[concat_node_name].node.input[index]
                )
                each_input_node = graph_info[each_input_name].node
                positive_input = False
                if each_input_node.op in ("Relu", "Relu6"):
                    positive_input = True
                else:
                    positive_input = g.has_positive_input(each_input_node.name)
                input_positive_status.append(positive_input)
            if not any(input_positive_status):  # pragma: no cover
                matched_nodes.remove(i)

    def _filter_unquantizable_concat_performance_only(self, matched_nodes):
        """OOB filter out unquantizable ConcatV2 OPs by checking the control flow rule."""
        target_concat_nodes = [i[0] for i in matched_nodes if i[-1][0] == "ConcatV2"]
        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper
        from neural_compressor.tensorflow.quantization.utils.utility import GraphAnalyzer

        g = GraphAnalyzer()
        g.graph = self.pre_optimized_model.graph_def
        graph_info = g.parse_graph()
        concat_nodes = g.query_fusion_pattern_nodes([["ConcatV2"]])
        for i in concat_nodes:
            concat_node_name = i[0]
            if concat_node_name not in target_concat_nodes:  # pragma: no cover
                continue
            input_positive_status = []
            control_flow = False
            for index in range(graph_info[concat_node_name].node.attr["N"].i):
                each_input_name = GraphRewriterHelper.node_name_from_input(
                    graph_info[concat_node_name].node.input[index]
                )
                each_input_node = graph_info[each_input_name].node
                if each_input_node.op in ("Switch"):  # pragma: no cover
                    control_flow = True
            if control_flow:  # pragma: no cover
                matched_nodes.remove(i)

    def parse_quant_config(self, quant_config, model, calib_iteration):
        """Parse the quant_config to tune_cfg.

        Args:
            quant_config: a quantization configuration.
            model: the fp32 model to be quantized.
            calib_iteration: the number of iteration for calibration.

        Returns:
            tune_cfg: a dict composed by necessary information for quantization.
        """
        capability = self._query_fw_capability(model)
        config_converter = TensorflowConfigConverter(quant_config, capability)
        tune_cfg = config_converter.parse_to_tune_cfg()
        tune_cfg["calib_sampling_size"] = self.calib_sampling_size
        tune_cfg["calib_iteration"] = calib_iteration
        tune_cfg["approach"] = "post_training_static_quant"
        tune_cfg["recipe_cfgs"] = tune_cfg.get("recipe_cfgs", {})
        tune_cfg["trial_number"] = 1

        return tune_cfg

    def _query_fw_capability(self, model):
        """Collect the model-wise and op-wise configuration for quantization.

        Args:
            model (tf.compat.v1.GraphDef): model definition.

        Returns:
            [dict]: model-wise & op-wise configuration for quantization.
        """
        if self.pre_optimized_model is None:
            from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.pre_optimize import (
                PreOptimization,
            )

            self.pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
            self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model(self.itex_mode)
            model.graph_def = self.pre_optimized_model.graph_def

        self.exclude_node_names = self.pre_optimizer_handle.get_excluded_node_names()
        patterns = self.query_handler.generate_internal_patterns()
        bf16_patterns = self.query_handler.get_bf16_patterns()
        matched_nodes = self.pre_optimizer_handle.get_matched_nodes(patterns)
        matched_bf16_nodes = self.pre_optimizer_handle.get_matched_nodes(bf16_patterns)
        original_graph_node_name = [i.name for i in model.graph_def.node]
        matched_nodes = sorted(
            matched_nodes, reverse=True, key=lambda i: (original_graph_node_name.index(i[0]), len(i[-1]))
        )

        def check_match(patterns, input_pattern):
            for i in patterns:
                if input_pattern == [i for i in i.replace("+", " ").strip().split(" ") if i]:  # pragma: no cover
                    return True
            return False

        if (self.new_api and self.performance_only) or self.itex_mode or os.getenv("TF_FORCE_CONCAT_OPTS") == "1":
            self._filter_unquantizable_concat_performance_only(matched_nodes)
        else:
            self._filter_unquantizable_concat(matched_nodes)
        copied_matched_nodes = copy.deepcopy(matched_nodes)
        for i in copied_matched_nodes:
            if i[-1][0] in self.query_handler.get_op_types()["int8"]:
                continue

            if not self.pre_optimizer_handle.has_positive_input(i[0]) and not check_match(
                self.query_handler.get_fuse_patterns()["int8"], i[-1]
            ):
                matched_nodes.remove(i)

        del copied_matched_nodes

        copied_matched_nodes = copy.deepcopy(matched_bf16_nodes)
        for i in copied_matched_nodes:
            for j in matched_nodes:
                if i[0] == j[0] and i in matched_bf16_nodes:
                    matched_bf16_nodes.remove(i)

        del copied_matched_nodes

        self._query_quantizable_ops(matched_nodes)
        self._query_bf16_ops(matched_bf16_nodes)
        capability = {"optypewise": self.get_optype_wise_ability(), "recipes_ops": self.recipes_ops}
        capability["opwise"] = copy.deepcopy(self.quantizable_op_details)
        capability["opwise"].update(self.bf16_op_details)
        logger.debug("Dump framework quantization capability:")
        logger.debug(capability)

        return capability

    def quantize_input(self, model):
        """Quantize the model to be able to take quantized input.

        Remove graph QuantizedV2 op and move its input tensor to QuantizedConv2d
        and calculate the min-max scale.

        Args:
            model (tf.compat.v1.GraphDef): The model to quantize input

        Return:
            model (tf.compat.v1.GraphDef): The quantized input model
            scale (float): The scale for dataloader to generate quantized input
        """
        scale = None
        # quantize input only support tensorflow version > 2.1.0
        if version1_lt_version2(tf.version.VERSION, "2.1.0"):  # pragma: no cover
            logger.warning("Quantize input needs tensorflow 2.1.0 and newer.")
            return model, scale

        graph_def = model.as_graph_def()
        node_name_mapping = {}
        quantize_nodes = []
        for node in graph_def.node:
            node_name_mapping[node.name] = node
            if node.op == "QuantizeV2":
                quantize_nodes.append(node)

        target_quantize_nodes = []
        for node in quantize_nodes:
            # only support Quantizev2 input op Pad and Placeholder
            if (
                node_name_mapping[node.input[0]].op == "Pad"
                and node_name_mapping[node_name_mapping[node.input[0]].input[0]].op == "Placeholder"
            ) or node_name_mapping[node.input[0]].op == "Placeholder":
                target_quantize_nodes.append(node)
        assert len(target_quantize_nodes) == 1, "only support 1 QuantizeV2 from Placeholder"
        quantize_node = target_quantize_nodes[0]

        quantize_node_input = node_name_mapping[quantize_node.input[0]]
        quantize_node_outputs = [node for node in graph_def.node if quantize_node.name in node.input]

        from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper

        if quantize_node_input.op == "Pad":
            pad_node_input = node_name_mapping[quantize_node_input.input[0]]
            assert pad_node_input.op == "Placeholder", "only support Pad between QuantizeV2 and Placeholder"
            from tensorflow.python.framework import tensor_util

            paddings_tensor = tensor_util.MakeNdarray(
                node_name_mapping[quantize_node_input.input[1]].attr["value"].tensor
            ).flatten()

            quantize_node.input[0] = quantize_node_input.input[0]
            for conv_node in quantize_node_outputs:
                assert "Conv2D" in conv_node.op, "only support QuantizeV2 to Conv2D"

                GraphRewriterHelper.set_attr_int_list(conv_node, "padding_list", paddings_tensor)
            graph_def.node.remove(quantize_node_input)

        from tensorflow.python.framework import dtypes

        GraphRewriterHelper.set_attr_dtype(node_name_mapping[quantize_node.input[0]], "dtype", dtypes.qint8)

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
        max_value = max_node.attr["value"].tensor.float_val[0]
        min_value = min_node.attr["value"].tensor.float_val[0]
        scale = 127.0 / max(abs(max_value), abs(min_value))
        # remove QuantizeV2 node
        graph_def.node.remove(quantize_node)

        graph = tf.Graph()
        with graph.as_default():
            # use name='' to avoid 'import/' to name scope
            tf.import_graph_def(graph_def, name="")
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
                res[op[1]] = {"activation": self.quantizable_op_details[op][0]["activation"]}
                if "weight" in self.quantizable_op_details[op][0]:
                    res[op[1]]["weight"] = self.quantizable_op_details[op][0]["weight"]
        for op in self.bf16_op_details:
            if op[1] not in res:
                res[op[1]] = {"activation": {"dtype": ["bf16"]}, "weight": {"dtype": ["bf16"]}}
        return res


class Tensorflow_ITEXAdaptor(TensorFlowAdaptor):  # pragma: no cover
    """Tensorflow ITEX Adaptor Class."""

    def __init__(self, framework_specific_info):
        """Initialization.

        Args:
            framework_specific_info: framework specific information.
        """
        super().__init__(framework_specific_info)

    @dump_elapsed_time("Pass quantize model")
    def quantize(
        self,
        quant_config: StaticQuantConfig,
        model: BaseModel,
        calib_dataloader: Callable = None,
        calib_iteration: int = 100,
        calib_func: Callable = None,
    ):
        """Execute the quantize process on the specified model.

        Args:
            quant_config: a quantization configuration.
            model: the fp32 model to be quantized.
            calib_dataloader: a data loader for calibration.
            calib_iteration: the iteration of calibration.
            calib_func: the function used for calibration, should be a substitution for calib_dataloader
            when the built-in calibration function of INC does not work for model inference.

        Returns:
            converted_model: the quantized INC model wrapper.
        """
        self.calib_sampling_size = calib_dataloader.batch_size * calib_iteration
        tune_cfg = self.parse_quant_config(quant_config, model, calib_iteration)
        self._tuning_cfg_to_fw(tune_cfg)
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        from neural_compressor.tensorflow.quantization.utils.graph_converter import GraphConverter

        self.calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
        if isinstance(calib_dataloader, BaseDataLoader):
            batch_size = calib_dataloader.batch_size
            try:
                for i in range(batch_size):
                    if self.calib_sampling_size % (batch_size - i) == 0:
                        calib_batch_size = batch_size - i
                        if i != 0:  # pragma: no cover
                            logger.warning(
                                "Reset `calibration.dataloader.batch_size` field "
                                "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                "divisible exactly by batch size"
                            )
                        break
                tmp_iterations = int(math.ceil(self.calib_sampling_size / calib_batch_size))
                calib_dataloader.batch(calib_batch_size)
                self.quantize_config["calib_iteration"] = tmp_iterations

                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=calib_dataloader,
                    calib_func=calib_func,
                    itex_mode=self.itex_mode,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
            except Exception:  # pragma: no cover
                from neural_compressor.tensorflow.quantization.utils.utility import get_model_input_shape

                batch_size = get_model_input_shape(model)
                logger.warning(
                    "Fail to forward with batch size={}, set to {} now.".format(calib_dataloader.batch_size, batch_size)
                )
                calib_dataloader.batch(batch_size)
                self.quantize_config["calib_iteration"] = self.calib_sampling_size
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=calib_dataloader,
                    itex_mode=self.itex_mode,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
        else:  # pragma: no cover
            if hasattr(calib_dataloader, "batch_size") and self.calib_sampling_size % calib_dataloader.batch_size != 0:
                iter = self.quantize_config["calib_iteration"]
                logger.warning(
                    "Please note that calibration sampling size {} "
                    "isn't divisible exactly by batch size {}. "
                    "So the real sampling size is {}.".format(
                        self.calib_sampling_size, calib_dataloader.batch_size, calib_dataloader.batch_size * iter
                    )
                )
            converted_model = GraphConverter(
                model,
                qt_config=self.quantize_config,
                recipes=self.recipes,
                int8_sequences=self.op_wise_sequences,
                fp32_ops=self.fp32_ops,
                bf16_ops=self.bf16_ops,
                data_loader=calib_dataloader,
                calib_func=calib_func,
                itex_mode=self.itex_mode,
                qdq_enabled=self.qdq_enabled,
                new_api=self.new_api,
                performance_only=self.performance_only,
                use_bf16=self.use_bf16,
            ).convert()

        self._dump_model_op_stats(converted_model.graph_def)

        return converted_model


class TensorFlowConfig:
    """Base config class for TensorFlow."""

    def __init__(self, precisions=None):
        """Init an TF object."""
        self._precisions = precisions

    @property
    def precisions(self):
        """Get precision."""
        return self._precisions

    @precisions.setter
    def precisions(self, precisions):  # pragma: no cover
        """Set precision."""
        if not isinstance(precisions, list):
            precisions = [precisions]
        for pr in precisions:
            self.check_value("precisions", pr, str, ["int8", "uint8", "fp32", "bf16", "fp16"])
        self._precisions = precisions

    @staticmethod
    def check_value(name, src, supported_type, supported_value=[]):  # pragma: no cover
        """Check if the given object is the given supported type and in the given supported value.

        Example::
            def datatype(self, datatype):
                if self.check_value("datatype", datatype, list, ["fp32", "bf16", "uint8", "int8"]):
                    self._datatype = datatype
        """
        if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
            assert False, "Type of {} items should be {} but not {}".format(
                name, str(supported_type), [type(i) for i in src]
            )
        elif not isinstance(src, list) and not isinstance(src, supported_type):
            assert False, "Type of {} should be {} but not {}".format(name, str(supported_type), type(src))

        if len(supported_value) > 0:
            if isinstance(src, str) and src not in supported_value:
                assert False, "{} is not in supported {}: {}. Skip setting it.".format(src, name, str(supported_value))
            elif (
                isinstance(src, list)
                and all([isinstance(i, str) for i in src])
                and any([i not in supported_value for i in src])
            ):
                assert False, "{} is not in supported {}: {}. Skip setting it.".format(src, name, str(supported_value))

        return True


class TensorflowQuery:
    """Tensorflow Query Capability Class."""

    def __init__(self, local_config_file=None, performance_only=False, itex_mode=False, quant_mode="static"):
        """Initialization.

        Args:
            local_config_file: local configuration file name.
            performance_only: oob performance only mode.
            itex_mode: check if itex mode.
            quant_mode: quantization mode, static or dynamic.
        """
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self.performance_only = performance_only
        self.quant_mode = quant_mode
        self.itex_mode = itex_mode
        self._one_shot_query()

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.

        If there's no matched configuration in the input yaml, we'll
        use the configuration of the nearest framework version field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        from functools import cmp_to_key

        config = None

        def _compare(version1, version2):
            if parse_version(version1) == parse_version(version2):  # pragma: no cover
                return 0
            elif parse_version(version1) < parse_version(version2):
                return -1
            else:
                return 1

        fallback_list = []
        for sub_data in data:
            if "default" in sub_data["version"]["name"]:
                assert config is None, "Only one default config " "is allowed in framework yaml file."
                config = sub_data

            if self.version in sub_data["version"]["name"]:
                return sub_data
            else:
                if sub_data["version"]["name"] == [
                    "2.11.0202242",
                    "2.11.0202250",
                    "2.11.0202317",
                    "2.11.0202323",
                    "2.14.0202335",
                    "2.14.dev202335",
                    "2.15.0202341",
                ]:
                    continue
                sorted_list = copy.deepcopy(sub_data["version"]["name"])
                sorted_list.remove("default") if "default" in sorted_list else None
                if isinstance(sorted_list, list):
                    # TensorFlow 1.15.0-up1/up2/up3 release versions are abnoraml release naming
                    # convention. Replacing them with dot for version comparison.
                    sorted_list = [i.replace("-up", ".") for i in sorted_list]
                    sorted_list = sorted(sorted_list, key=cmp_to_key(_compare), reverse=True)
                else:  # pragma: no cover
                    assert isinstance(sorted_list, str)
                    sorted_list = list(sorted_list.replace("-up", ".").split())
                for i in sorted_list:
                    if parse_version(self.version) >= parse_version(i):
                        fallback_list.append([i, sub_data])
                        break

        assert config is not None, "The default config in framework yaml must exist."
        nearest_version = str(0)
        for fallback in fallback_list:
            if parse_version(fallback[0]) > parse_version(nearest_version):
                nearest_version = fallback[0]
                config = fallback[1]

        return config

    def _one_shot_query(self):
        """One short query for some patterns."""
        # pylint: disable=E1136
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
                if not self.performance_only:
                    remove_int8_ops = [
                        "FusedBatchNorm",
                        "FusedBatchNormV2",
                        "FusedBatchNormV3",
                        "_MklFusedInstanceNorm",
                    ]
                    for op_type in remove_int8_ops:
                        while op_type in self.cur_config["int8"][self.quant_mode].keys():
                            self.cur_config["int8"][self.quant_mode].pop(op_type, None)

            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError(
                    "Please check if the format of {} follows Neural Compressor yaml schema.".format(self.cfg)
                )
        self._update_cfg_with_usr_definition()

    def _update_cfg_with_usr_definition(self):
        """Add user defined precision configuration."""
        tensorflow_config = TensorFlowConfig()
        if tensorflow_config.precisions is not None:  # pragma: no cover
            self.cur_config["precisions"]["names"] = ",".join(tensorflow_config.precisions)

    def get_version(self):
        """Get the current backend version information.

        Returns:
            [string]: version string.
        """
        return self.cur_config["version"]["name"]

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return {
            "int8": self.get_op_types_by_precision("int8"),
            "uint8": self.get_op_types_by_precision("uint8"),
            "bf16": self.get_op_types_by_precision("bf16"),
        }

    def get_fuse_patterns(self):
        """Get supported patterns by low precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the supported patterns.
        """
        spr_int8_pattern_list = [
            "Conv2D + BiasAdd",
            "Conv2D + BiasAdd + Add + Relu6 + Mul + Mul",
            "Conv2D + Add + Relu6 + Mul + Mul",
            "Conv2D + BiasAdd + swish_f32",
            "Conv2D + Add + swish_f32",
            "Conv2D + AddV2 + swish_f32",
            "Conv2D + swish_f32",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + Relu",
            "Conv2D + BiasAdd + Elu",
            "Conv2D + Elu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd + LeakyRelu",
            "Conv2D + BiasAdd + Add + LeakyRelu",
            "Conv2D + BiasAdd + AddV2 + LeakyRelu",
            "Conv2D + Add + LeakyRelu",
            "Conv2D + AddV2 + LeakyRelu",
            "Conv2D + LeakyRelu",
            "Conv2D + BiasAdd + Sigmoid",
            "Conv2D + Sigmoid",
            "Conv2D + BiasAdd + LeakyRelu + AddV2",
            "Conv2D + BiasAdd + LeakyRelu + Add",
            "Conv2D + LeakyRelu + AddV2",
            "Conv2D + LeakyRelu + Add",
            "Conv2D + BiasAdd + Relu + AddV2",
            "Conv2D + BiasAdd + Relu + Add",
            "Conv2D + Relu + AddV2",
            "Conv2D + Relu + Add",
            "Conv2D + Add",
            "Conv2D + AddV2",
            "Conv2D + AddV2 + Add",
            "Conv2D + Add + Add",
            "Conv2D + BiasAdd + Add",
            "Conv3D + Add",
            "Conv3D + AddV2",
            "Conv3D + BiasAdd",
            "Conv3D + BiasAdd + Add",
            "Conv3D + BiasAdd + AddV2",
            "Conv3D + AddV2 + AddV2",
            "DepthwiseConv2dNative + BiasAdd + Add + Relu6 + Mul + Mul",
            "DepthwiseConv2dNative + Add + Relu6 + Mul + Mul",
            "DepthwiseConv2dNative + BiasAdd + swish_f32",
            "DepthwiseConv2dNative + Add + swish_f32",
            "DepthwiseConv2dNative + AddV2 + swish_f32",
            "DepthwiseConv2dNative + swish_f32",
            "DepthwiseConv2dNative + BiasAdd + LeakyRelu",
            "DepthwiseConv2dNative + LeakyRelu",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "FusedBatchNormV3 + Relu",
            "FusedBatchNormV3 + LeakyRelu",
            "_MklFusedInstanceNorm + Relu",
            "_MklFusedInstanceNorm + LeakyRelu",
            "Conv2DBackpropInput + BiasAdd",
            "Conv3DBackpropInputV2 + BiasAdd",
        ]

        spr_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "Conv2D + Add + Add + Relu",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd",
            "MatMul + BiasAdd + Add",
            "MatMul + BiasAdd + AddV2",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd + Relu6",
            "MatMul + BiasAdd + LeakyRelu",
            "MatMul + BiasAdd + Gelu",
            "MatMul + BiasAdd + Elu",
            "MatMul + BiasAdd + Tanh",
            "MatMul + BiasAdd + Sigmoid",
            "MatMul + Add",
            "MatMul + AddV2",
            "MatMul + Relu",
            "MatMul + Relu6",
            "MatMul + LeakyRelu",
            "MatMul + Gelu",
            "MatMul + Elu",
            "MatMul + Tanh",
            "MatMul + Sigmoid",
            "BatchMatMul + Mul",
            "BatchMatMulV2 + Mul",
            "BatchMatMul + Add",
            "BatchMatMulV2 + Add",
            "BatchMatMul + AddV2",
            "BatchMatMulV2 + AddV2",
            "BatchMatMul + Mul + Add",
            "BatchMatMulV2 + Mul + Add",
            "BatchMatMul + Mul + AddV2",
            "BatchMatMulV2 + Mul + AddV2",
            "Conv3D + AddV2 + AddV2 + Relu",
            "Conv3D + Add + Relu",
            "Conv3D + AddV2 + Relu",
            "Conv3D + Relu",
            "Conv3D + Relu6",
            "Conv3D + Add + Relu6",
            "Conv3D + AddV2 + Relu6",
            "Conv3D + Elu",
            "Conv3D + LeakyRelu",
            "Conv3D + BiasAdd + Relu",
            "Conv3D + BiasAdd + Relu6",
            "Conv3D + BiasAdd + Elu",
            "Conv3D + BiasAdd + LeakyRelu",
            "Conv3D + Add + Elu",
            "Conv3D + Add + LeakyRelu",
            "Conv2DBackpropInput + BiasAdd",
            "Conv3DBackpropInputV2 + BiasAdd",
        ]

        tf_int8_pattern_list = ["Conv2D + BiasAdd", "Conv2D + BiasAdd + Relu", "Conv2D + BiasAdd + Relu6"]
        tf_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]
        tf1_15_up3_int8_pattern_list = [
            "Conv2D + BiasAdd",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + LeakyRelu",
            "Conv2D + BiasAdd + LeakyRelu + AddV2",
            "Conv2D + BiasAdd + Relu6",
        ]
        tf1_15_up3_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]
        old_tf_int8_pattern_list = ["MatMul + BiasAdd + Relu", "MatMul + BiasAdd"]
        old_tf_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]

        for index, pattern in enumerate(spr_int8_pattern_list):
            spr_int8_pattern_list[index] = "Dequantize + " + pattern + " + QuantizeV2"
        for index, pattern in enumerate(spr_uint8_pattern_list):
            spr_uint8_pattern_list[index] = "Dequantize + " + pattern + " + QuantizeV2"

        if not self.performance_only:
            remove_int8_ops = ["FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3", "_MklFusedInstanceNorm"]
            for op_type in remove_int8_ops:
                patterns = [
                    f"Dequantize + {op_type} + Relu + QuantizeV2",
                    f"Dequantize + {op_type} + LeakyRelu + QuantizeV2",
                ]
                for pattern in patterns:
                    while pattern in spr_int8_pattern_list:
                        spr_int8_pattern_list.remove(pattern)
                    while pattern in spr_uint8_pattern_list:
                        spr_uint8_pattern_list.remove(pattern)

        patterns = {}
        if tf.version.VERSION in spr_base_verions or self.itex_mode:
            patterns["int8"] = spr_int8_pattern_list
            patterns["uint8"] = spr_uint8_pattern_list
        elif version1_gte_version2(tf.version.VERSION, "2.1.0"):
            patterns["int8"] = tf_int8_pattern_list
            patterns["uint8"] = tf_uint8_pattern_list
            if self.itex_mode:  # pragma: no cover
                patterns["int8"].append("FusedBatchNormV3 + Relu")
                patterns["int8"].append("FusedBatchNormV3 + LeakyRelu")
        elif version1_eq_version2(tf.version.VERSION, "1.15.0-up3"):  # pragma: no cover
            patterns["int8"] = tf1_15_up3_int8_pattern_list
            patterns["uint8"] = tf1_15_up3_uint8_pattern_list
        else:  # pragma: no cover
            patterns["int8"] = old_tf_int8_pattern_list
            patterns["uint8"] = old_tf_uint8_pattern_list

        return patterns

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        for op_type, _ in self.cur_config["int8"][self.quant_mode].items():
            self.cur_config["int8"][self.quant_mode][op_type]["activation"]["quant_mode"] = self.quant_mode
        return self.cur_config["int8"][self.quant_mode]

    def get_op_types_by_precision(self, precision):
        """Get op types per precision.

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in ("bf16", "uint8", "int8")

        if precision == "int8":
            if tf.version.VERSION in spr_base_verions or self.itex_mode:
                op_type_list = [key for key in self.cur_config["int8"][self.quant_mode].keys()]
                if not self.performance_only and not self.itex_mode:
                    remove_int8_ops = [
                        "FusedBatchNorm",
                        "FusedBatchNormV2",
                        "FusedBatchNormV3",
                        "_MklFusedInstanceNorm",
                    ]
                    for op_type in remove_int8_ops:
                        while op_type in op_type_list:
                            op_type_list.remove(op_type)
                return op_type_list
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool"]
            return ["MatMul", "ConcatV2", "MaxPool", "AvgPool"]  # pragma: no cover
        if precision == "uint8":
            if tf.version.VERSION in spr_base_verions:
                return [key for key in self.cur_config["int8"][self.quant_mode].keys() if "Norm" not in key]
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool", "DepthwiseConv2dNative"]
            return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool"]  # pragma: no cover
        if precision == "bf16":
            if tf.version.VERSION in spr_base_verions:
                return self.cur_config[precision]
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return self.cur_config[precision]
            return []  # pragma: no cover

    def get_mixed_precision_combination(self):
        """Get the valid mixed precisions.

        Returns:
            [string list]: valid precision list.
        """
        if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(tf.version.VERSION, "1.15.0-up3"):
            return ["int8", "uint8", "bf16", "fp32"]
        return ["uint8", "fp32"]

    def get_bf16_patterns(self):
        """Get BF16 pattern list.

        Returns:
            [List]: bf16 pattern list.
        """
        bf16_op_types = [i for i in self.get_op_types_by_precision("bf16")]
        res = []
        for i in bf16_op_types:
            res.append([[i]])

        return res

    def get_eightbit_patterns(self, qdq_enabled=False):
        """Get eightbit op wise sequences information.

        Returns:
            [dictionary]: key is the op type while value is the list of sequences start
                        with the op type same as key value.
        """
        quantizable_op_types = self.get_op_types_by_precision("int8") + self.get_op_types_by_precision("uint8")
        int8_patterns = [
            i.replace("+", " ").split()
            for i in list(set(self.get_fuse_patterns()["int8"] + self.get_fuse_patterns()["uint8"]))
        ]
        res = {}
        for i in quantizable_op_types:
            if qdq_enabled:
                res[i] = [["Dequantize", i, "QuantizeV2"]]
            else:
                res[i] = [[i]]

        for pattern in int8_patterns:
            if qdq_enabled:
                op_type = pattern[1]
            else:
                op_type = pattern[0]
            if op_type in res:
                res[op_type].append(pattern)

        return res

    def generate_internal_patterns(self):
        """Translate the patterns defined in the yaml to internal pattern expression."""

        def _generate_pattern(data):
            length = [len(i) for i in data]
            res = []
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
                if len(value) >= last_len:
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
        for _, op_level_sequences in op_level_sequences.items():
            for similar_sequences in op_level_sequences:
                final_out.append(_generate_pattern(similar_sequences))

        return final_out


class TensorflowConfigConverter:
    """Convert `StaticQuantConfig` to the format used by static quant algo."""

    unify_op_type_mapping = UNIFY_OP_TYPE_MAPPING

    def __init__(self, quant_config: StaticQuantConfig, capability: Dict):
        """Init parser for TF static quant config.

        Args:
            quant_config: the keras static quant config.
            capability: the supported config lists for each op.
        """
        self.quant_config = quant_config
        self.capability = capability

    def update_opwise_config(self):
        """Update op-wise config.

        Args:
            quant_config: the Tensorflow static quant config.
        """
        op_wise_config = {}
        for op_name, op_config in self.quant_config.items():
            op_key_name = (op_name[0], self.unify_op_type_mapping[op_name[1]])
            if op_key_name not in self.capability["opwise"]:
                continue
            single_op_cap = self.capability["opwise"][op_key_name][0]
            single_op_config = {"activation": {}}

            single_op_config["activation"]["dtype"] = (
                op_config.act_dtype
                if op_config.act_dtype in single_op_cap["activation"]["dtype"]
                or op_config.act_dtype in ("fp32", "bf16")
                else single_op_cap["activation"]["dtype"][0]
            )

            single_op_config["activation"]["scheme"] = "sym" if op_config.act_sym else "asym"
            if single_op_config["activation"]["scheme"] not in single_op_cap["activation"]["scheme"]:
                single_op_config["activation"]["scheme"] = single_op_cap["activation"]["scheme"][0]

            single_op_config["activation"]["granularity"] = (
                op_config.act_granularity
                if op_config.act_granularity in single_op_cap["activation"]["granularity"]
                else single_op_cap["activation"]["granularity"][0]
            )

            single_op_config["activation"]["algorithm"] = (
                op_config.act_algorithm
                if op_config.act_algorithm in single_op_cap["activation"]["algorithm"]
                else single_op_cap["activation"]["algorithm"][0]
            )

            if "weight" not in single_op_cap:
                op_wise_config.update({op_key_name: single_op_config})
                continue

            single_op_config["weight"] = {}
            single_op_config["weight"]["dtype"] = (
                op_config.weight_dtype
                if op_config.weight_dtype in single_op_cap["weight"]["dtype"]
                or op_config.weight_dtype in ("fp32", "bf16")
                else single_op_cap["weight"]["dtype"][0]
            )

            single_op_config["weight"]["scheme"] = "sym" if op_config.weight_sym else "asym"
            if single_op_config["weight"]["scheme"] not in single_op_cap["weight"]["scheme"]:
                single_op_config["weight"]["scheme"] = single_op_cap["weight"]["scheme"][0]

            single_op_config["weight"]["granularity"] = (
                op_config.weight_granularity
                if op_config.weight_granularity in single_op_cap["weight"]["granularity"]
                else single_op_cap["weight"]["granularity"][0]
            )

            single_op_config["weight"]["algorithm"] = (
                op_config.weight_algorithm
                if op_config.weight_algorithm in single_op_cap["weight"]["algorithm"]
                else single_op_cap["weight"]["algorithm"][0]
            )

            op_wise_config.update({op_key_name: single_op_config})

        return op_wise_config

    def parse_to_tune_cfg(self):
        """The function that parses StaticQuantConfig to keras tuning config."""
        op_wise_config = self.update_opwise_config()
        tune_cfg = {"op": op_wise_config}

        return tune_cfg

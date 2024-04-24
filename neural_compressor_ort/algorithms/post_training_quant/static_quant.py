#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Quantizer for onnx models."""

import copy
import logging
import os

import numpy as np
import onnx
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnx import shape_inference
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization.quant_utils import QuantFormat
from neural_compressor_ort.algorithms.post_training_quant.base_quantizer import Quantizer
from neural_compressor_ort.algorithms.post_training_quant.operators import OPERATORS
from neural_compressor_ort.algorithms.post_training_quant.utils import (
    QuantizedInitializer,
    QuantizedValue,
    QuantizedValueType,
    _get_qrange_for_qType,
    find_by_name,
    get_node_original_name,
    make_dquant_node,
    make_quant_node,
    quantize_data_per_channel,
)
from neural_compressor_ort.utils.onnx_model import ONNXModel
#from neural_compressor.adaptor.ox_utils.operators import OPERATORS
#from neural_compressor.adaptor.ox_utils.util import (
#    QuantizedInitializer,
#    QuantizedValue,
#    QuantizedValueType,
#    ValueInfo,
#    __producer__,
#    __version__,
#    _get_qrange_for_qType,
#    cast_tensor,
#    dtype_mapping,
#    dtype_to_name,
#    find_by_name,
#    get_node_original_name,
#    make_dquant_node,
#    make_quant_node,
#    quantize_data,
#    quantize_data_per_channel,
#    support_pair,
#)
#from neural_compressor.model.onnx_model import ONNXModel

logger = logging.getLogger("neural_compressor")


class StaticQuantizer(Quantizer):
    """Static quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        quant_format=QuantFormat.QOperator,
        quantization_params={},
        op_types_to_quantize=[],
        fallback_list=["fp32"],
        reduce_range=None,
        add_qdq_pair_to_weight=False,
        optypes_to_exclude_output_quant=[],
        dedicated_qdq_pair=False,
        backend="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            mode (QuantizationMode): quantizaion mode
            static (bool): static or not
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            optypes_to_exclude_output_quant (list, optional): optypes to exclude output quantization. Defaults to [].
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            backend (str, optional): backend of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        super().__init__(
            mode="static_quant",
            model=model,
            q_config=q_config,
            static=True,
            quantization_params=quantization_params,
            op_types_to_quantize=op_types_to_quantize,
            )
        # self.model = ONNXModel(model) if not isinstance(model, ONNXModel) else model
        # model = (
        #     onnx.shape_inference.infer_shapes(self.model.model) if not self.model.is_large_model else self.model.model
        # )
        # self.config = q_config
        self.backend = backend
        self.reduce_range = reduce_range
        self.static = True  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        self.quantization_params = quantization_params
        self.op_types_to_quantize = op_types_to_quantize
        self.fallback_list = fallback_list
        self.new_nodes = []
        self.quant_format = "qoperator" if quant_format.value == 0 else "qdq"

        self.opset_version = self.check_opset_version()
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})
        self.replace_input = []
        self.remove_nodes = []
        # List of quantized weights
        self.quantized_value_map = {}
        self.new_value_info = {}

        # List of recalculated quantize weight for Gather op.
        self.recalculate_quantized_value = []

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (represented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        if not self.static:
            self.op_types_to_exclude_output_quantization = op_types_to_quantize
        else:
            self.op_types_to_exclude_output_quantization = optypes_to_exclude_output_quant

        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.dedicated_qdq_pair = dedicated_qdq_pair
        if self.opset_version < 13 and self.quant_format == "qdq":
            logger.warning(
                "Per-channel support with QDQ format requires opset version >= 13,"
                " use per-tensor granularity instead"
            )

    def should_convert(self, node):
        """Check if node should be converted."""
        name = get_node_original_name(node)
        if (
            name in self.config
            and self.config[name] not in self.fallback_list
            and self.quant_format != "qdq"
        ):
            return True
        else:
            return False

    def quantize_outputs(self, node, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node outputs."""
        if self.quant_format == "qdq":
            return
        for idx, tensor_name in enumerate(node.output):
            if (
                tensor_name in self.value_infos
                and self.value_infos[tensor_name].type.HasField("tensor_type")
                and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT # TODO
            ):
                return
            data_found = False
            refer_name = node.input[0] if direct_int8 else tensor_name

            if refer_name in self.quantized_value_map:
                scale_name = self.quantized_value_map[refer_name].scale_name
                zp_name = self.quantized_value_map[refer_name].zp_name
                data_found = True
            elif refer_name in self.quantization_params:
                data_found, scale_name, zp_name, _, _ = self._get_quantization_params(refer_name)

            if data_found is False:
                raise ValueError(
                    "Quantization parameters are not specified for param {}."
                    "In static mode quantization params for inputs and outputs "
                    "of nodes to be quantized are required.".format(tensor_name)
                )

            q_input = tensor_name
            q_output = tensor_name + "_quantized"
            dq_input = q_output
            dq_output = tensor_name + "_dequantized"

            quant_node_name = tensor_name + "_QuantizeLinear"
            dequant_node_name = tensor_name + "_DequantizeLinear"
            qlinear_node = make_quant_node(quant_node_name, [q_input, scale_name, zp_name], [q_output])
            dequant_node = make_dquant_node(dequant_node_name, [dq_input, scale_name, zp_name], [dq_output])
            self.new_nodes.extend([qlinear_node, dequant_node])
            
            if tensor_name not in self.quantized_value_map:
                quantized_value = QuantizedValue(tensor_name, dq_output, scale_name, zp_name, QuantizedValueType.Input)
                self.quantized_value_map[tensor_name] = quantized_value

    def quantize_inputs(self, node, indices=None, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""
        # Quantize the input
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if initializer.data_type != onnx_proto.TensorProto.FLOAT: # TODO
                    return

                dtype = (
                    self.config[node.name].weight_dtype
                    if initializer_use_weight_qType
                    else self.config[node.name].act_dtype
                )
                scheme = (
                    self.config[node.name].weight_sym
                    if initializer_use_weight_qType
                    else self.config[node.name].act_sym
                )
                weight = self._get_quantized_weight(initializer, dtype, scheme)
                self._update_weight(weight)

                if self.add_qdq_pair_to_weight and self.quant_format == "qdq":
                    node.input[idx] = weight.name
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"
                    qlinear_node = make_quant_node(
                        weight.name + "_QuantizeLinear",
                        [tensor_name, scale_name, zp_name],
                        [weight.name + "_quantized"],
                    )
                    dequant_node = make_dquant_node(
                        weight.name + "_DequantizeLinear",
                        [weight.name + "_quantized", scale_name, zp_name],
                        [weight.name + "_dequantized"],
                    )
                    self.new_nodes.extend([qlinear_node, dequant_node])

                else:
                    node.input[idx] = weight.name
                    q_weight_name = weight.name + "_quantized"
                    zp_name = weight.name + "_zero_point"
                    scale_name = weight.name + "_scale"

                    inputs = [q_weight_name, scale_name, zp_name]
                    output_name = tensor_name + "_DequantizeLinear"
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear", inputs, [weight.name + "_dequantized"], weight.name + "_DequantizeLinear"
                    )
                    self.new_nodes.append(dequant_node)

                self.replace_input.append([node, weight.name, dequant_node.output[0]])
                if weight.name not in self.quantized_value_map:
                    quantized_value = QuantizedValue(
                        weight.name, q_weight_name, scale_name, zp_name, QuantizedValueType.Initializer, None, dtype
                    )
                    self.quantized_value_map[weight.name] = quantized_value
            else:
                
                if (
                    tensor_name in self.value_infos
                    and self.value_infos[tensor_name].type.HasField("tensor_type")
                    and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT # TODO
                ):
                    return

                if tensor_name in self.quantized_value_map:
                    # node input is model input and it has been quantized, don't insert QDQ pair
                    if tensor_name in self.model.input():
                        continue
                    scale_name = self.quantized_value_map[tensor_name].scale_name
                    zp_name = self.quantized_value_map[tensor_name].zp_name
                    data_found = True
                else:
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

                if data_found is False:
                    raise ValueError(
                        "Quantization parameters are not specified for param {}."
                        "In static mode quantization params for inputs and outputs "
                        "of nodes to be quantized are required.".format(tensor_name)
                    )

                if direct_int8:
                    # direct int8 models will be quantized only if their inputs are quantized
                    if node.input[0] not in self.quantized_value_map:
                        return

                q_input = tensor_name
                q_output = tensor_name + "_quantized"
                dq_input = q_output
                dq_output = tensor_name + "_dequantized"
                self.replace_input.append([node, tensor_name, dq_output])

                quant_node_name = tensor_name + "_QuantizeLinear"
                dequant_node_name = tensor_name + "_DequantizeLinear"
                qlinear_node = make_quant_node(quant_node_name, [q_input, scale_name, zp_name], [q_output])
                dequant_node = make_dquant_node(dequant_node_name, [dq_input, scale_name, zp_name], [dq_output])
                self.new_nodes.extend([qlinear_node, dequant_node])

                if tensor_name not in self.quantized_value_map:
                    quantized_value = QuantizedValue(
                        tensor_name, dq_output, scale_name, zp_name, QuantizedValueType.Input
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
                
    def quantize_weights_per_channel(self, node, indices, weight_qType, scheme, axis):
        """Quantize weights per-channel."""
        if self.opset_version < 13 and self.quant_format == "qdq":
            self.quantize_inputs(node, indices)
            return

        for idx, inp in enumerate(node.input):
            if idx not in indices:
                continue

            if self.add_qdq_pair_to_weight and self.quant_format == "qdq":
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(inp, weight_qType, scheme, axis)
                weight_name = (
                    ("_").join([inp, str(weight_qType)]) if self.model.get_initializer_share_num(inp) > 1 else inp
                )
                qlinear_node = make_quant_node(
                    weight_name + "_QuantizeLinear",
                    [inp, scale_name, zp_name],
                    [q_name],
                    axis,
                )
                dequant_node = make_dquant_node(
                    weight_name + "_DequantizeLinear",
                    [q_name, scale_name, zp_name],
                    [weight_name + "_dequantized"],
                    axis,
                )
                node.input[idx] = weight_name
                self.replace_input.append([node, weight_name, dequant_node.output[0]])
                self.new_nodes.extend([qlinear_node, dequant_node])
            else:
                q_name, zp_name, scale_name = self.quantize_weight_per_channel(inp, weight_qType, scheme, axis)
                weight_name = (
                    ("_").join([inp, str(weight_qType)]) if self.model.get_initializer_share_num(inp) > 1 else inp
                )
                dequant_node = make_dquant_node(
                    weight_name + "_DequantizeLinear",
                    [q_name, scale_name, zp_name],
                    [weight_name + "_dequantized"],
                    axis,
                )
                self.new_nodes.append(dequant_node)
                node.input[idx] = weight_name

                # Replace weight_name with output of DequantizeLinear
                self.replace_input.append([node, weight_name, dequant_node.output[0]])

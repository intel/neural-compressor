# Copyright (c) 2023 Intel Corporation
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
"""Dynamic quantizer for onnx models."""

import copy
import logging
import os

import numpy as np
import onnx
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnx import shape_inference
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

logger = logging.getLogger("neural_compressor")


class DynamicQuantizer(Quantizer):
    """Dynamic quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        quantization_params={},
        op_types_to_quantize=[],
        fallback_list=["fp32"],
        reduce_range=None,
        optypes_to_exclude_output_quant=[],
        backend="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
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
            mode="dynamic_quant",
            model=model,
            q_config=q_config,
            static=False,
            quantization_params=quantization_params,
            op_types_to_quantize=op_types_to_quantize,
            )
        #self.model = ONNXModel(model) if not isinstance(model, ONNXModel) else model
        #model = (
        #    onnx.shape_inference.infer_shapes(self.model.model) if not self.model.is_large_model else self.model.model
        #)
        #self.config = q_config
        self.backend = backend
        self.reduce_range = reduce_range
        self.fuse_dynamic_quant = False
        #self.quantization_params = quantization_params
        #self.op_types_to_quantize = op_types_to_quantize
        self.fallback_list = fallback_list
        self.new_nodes = []

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
        self.op_types_to_exclude_output_quantization = op_types_to_quantize

    def should_convert(self, node):
        """Check if node should be converted."""
        name = get_node_original_name(node)
        if (
            name in self.config
            and self.config[name] not in self.fallback_list
        ):
            return True
        else:
            return False

    def quantize_inputs(self, node, indices=None, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""
        # Quantize the input
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if initializer.data_type != onnx_proto.TensorProto.FLOAT:
                    return
                if node.op_type not in self.op_types_to_quantize:
                    dtype = (
                        onnx_proto.TensorProto.INT8 if initializer_use_weight_qType else onnx_proto.TensorProto.UINT8
                    )
                    sym = True if initializer_use_weight_qType else False
                else:
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
                quantized_value = QuantizedValue(
                    weight.name, q_weight_name, scale_name, zp_name, QuantizedValueType.Initializer, None, dtype
                )
                if weight.name not in self.quantized_value_map:
                    self.quantized_value_map[weight.name] = quantized_value
            else:
                if (
                    tensor_name in self.value_infos
                    and self.value_infos[tensor_name].type.HasField("tensor_type")
                    and self.value_infos[tensor_name].type.tensor_type.elem_type != TensorProto.FLOAT
                ):
                    return

                if tensor_name in self.quantized_value_map:
                    scale_name = self.quantized_value_map[tensor_name].scale_name
                    zp_name = self.quantized_value_map[tensor_name].zp_name
                    data_found = True
                else:
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

                qlinear_node = self.model.find_node_by_name(
                    tensor_name + "_QuantizeLinear", self.new_nodes, self.model.graph()
                )
                if qlinear_node is None:
                    if (
                        self.fuse_dynamic_quant
                        and self.config[node.name].act_dtype == onnx_proto.TensorProto.UINT8
                        and not self.config[node.name].act_sym
                    ):
                        # DynamicQuantizeLinear supports uint8 input for CPU EP, supports uint8 and int8 for DML EP
                        scale_name = tensor_name + "_scale"
                        zeropoint_name = tensor_name + "_zero_point"
                        if find_by_name(scale_name, self.model.initializer()):
                            self.model.remove_initializer(find_by_name(scale_name, self.model.initializer()))
                        if find_by_name(zeropoint_name, self.model.initializer()):
                            self.model.remove_initializer(find_by_name(zeropoint_name, self.model.initializer()))
                        qlinear_node = onnx.helper.make_node(
                            "DynamicQuantizeLinear",
                            [tensor_name],
                            [tensor_name + "_dynamic_quantized", scale_name, zeropoint_name],
                            tensor_name + "_QuantizeLinear",
                        )
                    else:
                        scale_name, zp_name, _, _ = self._get_dynamic_input_quantization_params(
                            tensor_name, self.config[node.name].act_dtype
                        )
                        qlinear_node = make_quant_node(
                            tensor_name + "_QuantizeLinear",
                            [tensor_name, scale_name, zp_name],
                            [tensor_name + "_quantized"],
                        )
                    if qlinear_node not in self.new_nodes:
                        self.new_nodes.append(qlinear_node)
                    self.quantized_value_map[tensor_name] = QuantizedValue(
                        tensor_name,
                        qlinear_node.output[0],
                        scale_name,
                        zp_name,
                        self.config[node.name].act_dtype,
                    )
                self.replace_input.append([node, tensor_name, qlinear_node.output[0]])

    def quantize_weights_per_channel(self, node, indices, weight_qType, scheme, axis):
        """Quantize weights per-channel."""
        for idx, inp in enumerate(node.input):
            if idx not in indices:
                continue

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

    def _get_dynamic_input_quantization_params(self, input_name, qType):
        """Create nodes for dynamic quantization of input.

        Args:
            input_name (string): Name of the input.
            qType (int): type to quantize to.
        """
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name)

        return self._get_dynamic_input_quantization_params_uint8(input_name)

    def _get_dynamic_input_quantization_params_int8(self, input_name):  # pragma: no cover
        """Create nodes for dynamic quantization of input to int8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        self.new_nodes.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        self.new_nodes.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        self.new_nodes.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        initializer_div = onnx.helper.make_tensor(
            self.fixed_qrange_int8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [_get_qrange_for_qType(qType) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name):
        """Create nodes for dynamic quantization of input to uint8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        initializer_qrange = onnx.helper.make_tensor(
            self.fixed_qrange_uint8_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [_get_qrange_for_qType(qType)],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx_proto.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        self.new_nodes.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        self.new_nodes.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        self.new_nodes.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        self.new_nodes.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        self.new_nodes.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []
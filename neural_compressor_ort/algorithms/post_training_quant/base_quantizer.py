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
"""Base quantizer for onnx models."""

import copy
import logging
import os

import numpy as np
import onnx
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnx import shape_inference
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from neural_compressor_ort.algorithms.post_training_quant.operators import OPERATORS
from neural_compressor_ort.algorithms.post_training_quant.utils import (
    QuantizedInitializer,
    QuantizedValue,
    QuantizedValueType,
    ValueInfo,
    __producer__,
    __version__,
    _get_qrange_for_qType,
    cast_tensor,
    dtype_mapping,
    dtype_to_name,
    find_by_name,
    get_node_original_name,
    make_dquant_node,
    make_quant_node,
    quantize_data,
    quantize_data_per_channel,
    support_pair,
)
from neural_compressor.model.onnx_model import ONNXModel

logger = logging.getLogger("neural_compressor")


class Quantizer:
    """Quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        mode,
        static,
        quantization_params,
        op_types_to_quantize,
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
        self.model = ONNXModel(model) if not isinstance(model, ONNXModel) else model
        model = (
            onnx.shape_inference.infer_shapes(self.model.model) if not self.model.is_large_model else self.model.model
        )
        self.config = q_config
        self.backend = backend
        self.reduce_range = reduce_range
        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        self.quantization_params = quantization_params
        self.op_types_to_quantize = op_types_to_quantize
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

        if not self.static:
            self.op_types_to_exclude_output_quantization = op_types_to_quantize
        else:
            self.op_types_to_exclude_output_quantization = optypes_to_exclude_output_quant

        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.dedicated_qdq_pair = dedicated_qdq_pair

    def check_opset_version(self):
        """Check opset version."""
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if 1 != len(ai_onnx_domain):
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version > 10:
            self.fuse_dynamic_quant = True
        elif opset_version < 10:
            logger.warning(
                f"Warning: The original model opset version is {opset_version}, which does not support node "
                + "fusions. Please update the model to opset >= 11 for better performance."
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        return opset_version

    def should_quantize(self, node):
        """Check if node should be quantized."""
        if node.name in self.config and self.config[node.name] not in self.fallback_list:
            return True
        elif (
            get_node_original_name(node) in self.config
            and self.config[get_node_original_name(node)] not in self.fallback_list
        ):
            return True
        else:
            return False

    def quantize_model(self):
        """Quantize onnx model."""
        # step 1: insert q-dq, cast-cast pairs

        self.insert_qdq()

        # step 2: convert q-node-dq to qoperator format if needed
        self.convert_qdq_to_operator_oriented()

        self.model.remove_unused_nodes()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model


    def insert_qdq(self):
        """Insert Q/DQ pairs."""
        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = OPERATORS[self.mode][node.op_type](self, node)
                if op_quantizer.quantize_check():
                    op_quantizer.quantize()
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def convert_qdq_to_operator_oriented(self):
        """Convert QDQ to QOperator format."""
        self.new_nodes = []
        self.remove_nodes = []
        self.replace_input = []
        for node in self.model.nodes():
            if node.op_type not in ["QuantizeLinear", "DequantizeLinear"] and self.should_convert(node):
                op_converter = OPERATORS[self.mode][node.op_type](self, node)
                if op_converter.convert_check():
                    op_converter.convert()
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def quantize_inputs(self, node, indices=None, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""

    def quantize_bias_tensor(self, node):
        """Quantize bias."""
        input_name, weight_name, bias_name = node.input
        if (
            self.quantization_params is None
            or input_name not in self.quantization_params
            or input_name not in self.quantized_value_map
            or (
                input_name in self.quantized_value_map
                and find_by_name(self.quantized_value_map[input_name].scale_name, self.model.initializer()) is None
            )
        ):
            self._dynamic_quantize_bias(input_name, weight_name + "_scale", bias_name, bias_name + "_quantized")
        else:
            beta = 1.0
            if node.op_type in ["Gemm"]:
                beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
                if len(beta_attribute):
                    beta = onnx.helper.get_attribute_value(beta_attribute[0])
            _, quant_value = self.quantize_bias(bias_name, input_name, weight_name, beta)
            if self.model.get_initializer_share_num(bias_name) == 1:
                self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            axis = None
            if find_by_name(weight_name + "_DequantizeLinear", self.new_nodes):
                dq_node = find_by_name(weight_name + "_DequantizeLinear", self.new_nodes)
                if dq_node.op_type == "DequantizeLinear" and find_by_name("axis", dq_node.attribute):
                    axis = find_by_name("axis", dq_node.attribute).i
            dequant_node = make_dquant_node(bias_name + "_DequantizeLinear", inputs, [bias_name + "_dequantized"], axis)
            self.new_nodes.append(dequant_node)
            self.replace_input.append(
                [find_by_name(node.name, self.model.nodes()), bias_name, bias_name + "_dequantized"]
            )

    def quantize_bias(self, bias_name, input_name, weight_name, beta=1.0):
        """Quantized the bias.

        Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """
        # get scale for weight
        weight_scale_initializer = find_by_name(weight_name + "_scale", self.model.initializer())
        weight_scale = (
            self.tensor_proto_to_array(weight_scale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(weight_scale_initializer)
        )

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = (
            self.tensor_proto_to_array(bias_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(bias_initializer)
        )
        quantized_bias_name = bias_name + "_quantized"

        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError(f"Expected {input_name} to be in quantized value map for static quantization")
        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
        input_scale = (
            self.tensor_proto_to_array(inputscale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(inputscale_initializer)
        )

        # calculate scale for bias

        bias_scale = input_scale * weight_scale * beta

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

        # update bias initializer
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])

        # update scale initializer
        quantized_bias_scale_name = bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer().extend([packed_bias_scale_initializer])

        # update zero initializer
        quantized_bias_zp_name = bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        self.model.initializer().extend([packed_bias_zp_initializer])

        # log entries for this quantized bias value
        quantized_bias_entry = QuantizedInitializer(
            bias_name,
            bias_initializer,
            [0],
            [0],
            [0],
            [bias_scale],
            bias_data,
            quantized_data,
            qType=onnx_proto.TensorProto.INT32,
        )

        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            None,
            onnx_proto.TensorProto.INT32,
        )
        return quantized_bias_name, quantized_value

    def quantize_weight_per_channel(self, weight_name, weight_qType, scheme, channel_axis):
        """Quantize weight per-channel."""
        name = (
            ("_").join([weight_name, str(weight_qType)])
            if self.model.get_initializer_share_num(weight_name) > 1
            else weight_name
        )
        if name in self.quantized_value_map:
            return (name + "_quantized", name + "_zero_point", name + "_scale")

        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights = quantize_data_per_channel(
            weights, channel_axis, _get_qrange_for_qType(weight_qType, self.reduce_range), weight_qType, scheme
        )

        weight = QuantizedInitializer(
            name,
            initializer,
            rmin,
            rmax,
            zero_point,
            scale,
            weights,
            quantized_weights.flatten().tolist(),
            channel_axis,
            weight_qType,
        )

        self._update_weight(weight)
        quantized_value = QuantizedValue(
            weight.name,
            weight.name + "_quantized",
            weight.name + "_scale",
            weight.name + "_zero_point",
            QuantizedValueType.Initializer,
            None,
            weight_qType,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return (weight.name + "_quantized", weight.name + "_zero_point", weight.name + "_scale")

    def _update_weight(self, weight):
        """Update weight.

        Given a weight object, update the graph by doing the following:
         - remove old initializer, update new initializers for
           quantized weight, zero point, and scale
         - remove old weight input, update with new inputs for
           quantized weight, zero point, and scale
        This function does NOT update the nodes in the graph, just initializers and inputs
        """
        if weight.name in self.quantized_value_map:
            return
        packed_weight_name = weight.name + "_quantized"
        scale_name = weight.name + "_scale"
        zero_point_name = weight.name + "_zero_point"

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(
            weight.quantized_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight.qType]
        ).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if not self.add_qdq_pair_to_weight or self.mode != "qdq":
            self.model.initializer().append(packed_weight_initializer)
        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else:  # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, weight.scales
        )
        zero_initializer = onnx.helper.make_tensor(
            zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points
        )

        self.model.initializer().extend([scale_initializer, zero_initializer])

    @staticmethod
    def tensor_proto_to_array(initializer, base_dir=""):
        """Convert TensorProto to array."""
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer, base_dir)
        else:
            raise ValueError(
                "Only float type quantization is supported. \
                Weights {} is {}.".format(
                    initializer.name, dtype_to_name(dtype_mapping, initializer.data_type)
                )
            )
        return weights

    def _get_quantization_params(self, param_name):
        """Create initializers and inputs in the graph for zero point and scale of output.

        Zero point and scale values are obtained from self.quantization_params if specified.

        Args:
            param_name (string): Name of the quantization parameter.
        """
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""

        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError(
                "Quantization parameters should contain zero point and scale. "
                "Specified values for output {}: {}".format(param_name, params)
            )

        zero_point_values = [params[0]]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1]]
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantized_weight(self, initializer, qType, sym):
        """Get quantized weight."""
        name = (
            ("_").join([initializer.name, str(qType)])
            if self.model.get_initializer_share_num(initializer.name) > 1
            else initializer.name
        )
        if name in self.quantized_value_map:
            return self.quantized_value_map[name]
        weights_data = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(
            weights_data.flatten().tolist(), _get_qrange_for_qType(qType, self.reduce_range), qType, sym
        )
        weight = QuantizedInitializer(
            name,
            initializer,
            [rmin],
            [rmax],
            [zero_point],
            [scale],
            weights_data,
            quantized_weights_data,
            axis=None,
            qType=qType,
        )

        return weight

    def is_valid_quantize_weight(self, weight_name):
        """Check weight can be quantized."""
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type == onnx_proto.TensorProto.FLOAT
        else:
            return weight_name in self.quantized_value_map

    def get_bias_add_nodes(self, node, weight_name, last_output, quantized_bias_name):
        """Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node.

        Args:
            node (NodeProto): current node (Conv)
            weight_name (string): weight name
            last_output (_type_): output of previous node (input to bias add)
            quantized_bias_name (string): bias name
        """
        # Add tensors for the shape to be reshaped to
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is None:
            raise ValueError("Expected {} to be an initializer".format(node.input[1]))

        # Add reshape for correct broadcast
        reshape_input_data = quantized_bias_name
        reshape_input_shape = quantized_bias_name + "_reshape_shape"
        reshape_input = [reshape_input_data, reshape_input_shape]
        reshape_shape = np.ones((len(weight.dims)), dtype=np.int64)
        reshape_shape[1] = -1
        init_shape = onnx.helper.make_tensor(
            reshape_input_shape, onnx_proto.TensorProto.INT64, [len(weight.dims)], reshape_shape
        )
        self.model.add_initializer(init_shape)

        reshape_op_output = node.output[0] + "_reshape"
        reshape_node = onnx.helper.make_node(
            "Reshape", reshape_input, [reshape_op_output], quantized_bias_name + "reshape"
        )
        self.new_nodes.append(reshape_node)

        # Add an Add operation for bias
        bias_add_input = [last_output]
        bias_add_input.append(reshape_op_output)
        add_node_output = node.output[0] + "_bias_add"
        add_node = onnx.helper.make_node("Add", bias_add_input, [add_node_output], quantized_bias_name + "bias_add")
        self.new_nodes.append(add_node)
        return add_node_output

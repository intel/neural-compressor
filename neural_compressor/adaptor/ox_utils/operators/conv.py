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
"""Conv Operator."""


import onnx
from onnx import onnx_pb as onnx_proto

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, find_by_name


@op_registry(op_types="Conv, FusedConv")
class ConvOperator(Operator):
    """Conv Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(ConvOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        if node.op_type == "FusedConv":
            kwargs = {}
            for attribute in node.attribute:
                if attribute.name == "activation" and attribute.s in [b"Relu", b"Clip"]:
                    continue
                if attribute.name == "activation_params":
                    continue
                kwargs.update(attribute_to_kwarg(attribute))
            conv = onnx.helper.make_node("Conv", node.input, node.output, node.name, **kwargs)
            node.CopyFrom(conv)

        self.quantizer.quantize_inputs(node, [0])

        if self.per_channel:
            self.quantizer.quantize_weights_per_channel(node, [1], self.weight_dtype, self.weight_scheme, 0)
        else:
            self.quantizer.quantize_inputs(node, [1])

        if not self.disable_qdq_for_node_output or self.quantizer.mode != "qdq":
            self.quantizer.quantize_outputs(node)

        if len(node.input) == 3:
            self.quantizer.quantize_bias_tensor(node)

        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["dynamic", "static"], "convert format for {} should be in [dynamic, static]".format(
            node.op_type
        )
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node
        if convert_format == "dynamic":
            inputs = []
            parents = self.quantizer.model.get_parents(node)
            if parents[0].op_type == "QuantizeLinear":
                inputs.append(parents[0].output[0])
                inputs.append(parents[1].input[0])
                inputs.append(parents[0].input[2])
                inputs.append(parents[1].input[2])
                scale_0 = parents[0].input[1]
            else:
                inputs.append(parents[0].output[0])
                inputs.append(parents[1].input[0])
                inputs.append(parents[0].output[2])
                inputs.append(parents[1].input[2])
                scale_0 = parents[0].output[1]
            scale_1 = parents[1].input[1]
            # quantize bias if exist
            quantized_bias_name = ""
            bias_present = False
            if len(node.input) == 3:
                quantized_bias_name = node.input[2] + "_quantized"
                bias_present = True

            conv_integer_output = node.output[0] + "_output_quantized"

            kwargs = {}
            for attribute in node.attribute:
                if attribute.name == "activation" and attribute.s in [b"Relu", b"Clip"]:  # pragma: no cover
                    continue
                if attribute.name == "activation_params":  # pragma: no cover
                    continue
                kwargs.update(attribute_to_kwarg(attribute))
            conv_integer_node = onnx.helper.make_node("ConvInteger", inputs, [conv_integer_output], node.name, **kwargs)
            self.quantizer.new_nodes.append(conv_integer_node)

            # Add bias add nodes
            if bias_present:
                conv_integer_output = self.quantizer.get_bias_add_nodes(
                    node, parents[1].input[0], conv_integer_output, quantized_bias_name
                )

            # Add cast operation to cast convInteger output to float.
            cast_op_output = conv_integer_output + "_cast_output"
            cast_node = onnx.helper.make_node(
                "Cast",
                [conv_integer_output],
                [cast_op_output],
                conv_integer_output + "_cast",
                to=onnx_proto.TensorProto.FLOAT,
            )
            self.quantizer.new_nodes.append(cast_node)

            # Add mul operation to multiply scales of two inputs.
            scales_mul_op = node.name + "_scales_mul"

            scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
            if scales_mul_node is None:
                scales_mul_node = onnx.helper.make_node(
                    "Mul", [scale_0, scale_1], [scales_mul_op + ":0"], scales_mul_op
                )
                self.quantizer.new_nodes.append(scales_mul_node)

            scales_mul_op_output = scales_mul_node.output[0]

            # Add mul operation to multiply mul_scales_op result with output of ConvInteger
            # and make the output of this node the same as output of original conv node.
            output_scale_mul_op = node.name + "_output_scale_mul"
            self.quantizer.new_nodes.append(
                onnx.helper.make_node(
                    "Mul", [cast_op_output, scales_mul_op_output], [node.output[0]], output_scale_mul_op
                )
            )
            self.quantizer.remove_nodes.extend(parents[1:])
            self.quantizer.remove_nodes.append(node)
        elif convert_format == "static":
            if len(self.quantizer.model.get_children(node)) == 0 or not node.name.endswith(
                "_quant"
            ):  # pragma: no cover
                return
            parents = self.quantizer.model.get_parents(node)
            child = self.quantizer.model.get_children(node)[0]
            qlinear_conv_inputs = []
            for parent in parents[0:2]:
                qlinear_conv_inputs.extend(parent.input)
            qlinear_conv_inputs.extend(child.input[1:])
            if len(parents) == 3:
                qlinear_conv_inputs.append(parents[-1].input[0])

            qlinear_conv_output = child.output[0]

            kwargs = {}
            for attribute in node.attribute:
                if attribute.name == "activation" and attribute.s in [b"Relu", b"Clip"]:  # pragma: no cover
                    continue
                if attribute.name == "activation_params":  # pragma: no cover
                    continue
                kwargs.update(attribute_to_kwarg(attribute))

            qlinear_conv_node = onnx.helper.make_node(
                "QLinearConv", qlinear_conv_inputs, [qlinear_conv_output], node.name, **kwargs
            )
            self.quantizer.new_nodes.append(qlinear_conv_node)
            self.quantizer.remove_nodes.extend(parents)
            self.quantizer.remove_nodes.append(child)
            self.quantizer.remove_nodes.append(node)


@qop_registry(op_types="QLinearConv")
class QConvOperator(QOperator):
    """QLinearConv Operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        node = self.node
        add_nodes = []
        inits = []
        # input dq
        in_dq1 = onnx.helper.make_node(
            "DequantizeLinear", node.input[:3], [node.name + "_in_dequant1"], node.name + "_in_dequant1"
        )

        in_dq2 = onnx.helper.make_node(
            "DequantizeLinear", node.input[3:6], [node.name + "_in_dequant2"], node.name + "_in_dequant2"
        )

        add_nodes.extend([in_dq1, in_dq2])
        inputs = [node.name + "_in_dequant1", node.name + "_in_dequant2"]
        if len(node.input) == 9:
            import numpy as np

            input_scale = onnx.numpy_helper.to_array(find_by_name(node.input[1], self.initializers))
            weight_scale = onnx.numpy_helper.to_array(find_by_name(node.input[4], self.initializers))
            bias_scale = input_scale * weight_scale

            # update scale initializer
            bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
            bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, node.input[8] + "_scale")
            inits.extend([bias_scale_initializer])

            # update zero initializer
            bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
            bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, node.input[8] + "_zero_point")
            inits.extend([bias_zp_initializer])
            in_dq3 = onnx.helper.make_node(
                "DequantizeLinear",
                [node.input[8], bias_scale_initializer.name, bias_zp_initializer.name],
                [node.name + "_in_dequant3"],
                node.name + "_in_dequant3",
            )
            inputs.append(in_dq3.name)
            add_nodes.append(in_dq3)
        # output q
        out_q = onnx.helper.make_node(
            "QuantizeLinear", [node.name + "_out", node.input[6], node.input[7]], node.output, node.name + "_out_quant"
        )
        outputs = [node.name + "_out"]
        add_nodes.append(out_q)

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        binary_node = onnx.helper.make_node(
            node.op_type.split("QLinear")[-1], inputs, outputs, node.name + "_convert", **kwargs
        )
        add_nodes.append(binary_node)
        return True, add_nodes, inits

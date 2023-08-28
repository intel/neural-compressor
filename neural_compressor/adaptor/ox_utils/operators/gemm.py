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
"""Gemm Operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, find_by_name, is_B_transposed, ms_domain


@op_registry(op_types="Gemm")
class GemmOperator(Operator):
    """Gemm Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(GemmOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if len(node.input) == 3 and not find_by_name(node.input[2], self.quantizer.model.initializer()):
            from neural_compressor.utils import logger

            logger.warning(
                "Bias of Gemm node '{}' is not constant. "
                "Exclude this node can get better performance.".format(node.name)
            )
            if self.quantizer.mode != "qdq":
                return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(
                node, [1], self.weight_dtype, self.weight_scheme, 0 if is_B_transposed(node) else 1
            )
        else:
            self.quantizer.quantize_inputs(node, [1])

        if len(node.input) == 3 and find_by_name(node.input[2], self.quantizer.model.initializer()):
            self.quantizer.quantize_bias_tensor(node)
            beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
            if len(beta_attribute):
                beta_attribute[0].f = 1.0

        if not self.disable_qdq_for_node_output or self.quantizer.mode != "qdq":
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["static"], "convert format for {} should be in ['static']".format(node.op_type)

        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        child = self.quantizer.model.get_children(node)[0]
        qgemm_output = child.output[0]
        qgemm_inputs = []
        for parent in parents[:-1]:
            qgemm_inputs.extend(parent.input)
        qgemm_inputs.append(parents[-1].input[0])
        qgemm_inputs.extend(child.input[1:])

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name != "beta":
                kwargs.update(attribute_to_kwarg(attribute))
                kwargs["domain"] = ms_domain

        qgemm_node = onnx.helper.make_node("QGemm", qgemm_inputs, [qgemm_output], node.name, **kwargs)

        self.quantizer.new_nodes.append(qgemm_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(node)


@qop_registry(op_types="QGemm")
class QGemmOperator(QOperator):
    """QGemm Operator."""

    def __init__(self, onnx_node, children, initializers):
        """Initialization."""
        super().__init__(onnx_node, children, initializers)

    def convert(self):
        """Convert to QDQ format."""
        import numpy as np

        node = self.node
        add_nodes = []
        inits = []

        input_scale = onnx.numpy_helper.to_array(find_by_name(node.input[1], self.initializers))
        weight_scale = onnx.numpy_helper.to_array(find_by_name(node.input[4], self.initializers))
        bias_scale = input_scale * weight_scale

        # input dq
        in_dq1 = onnx.helper.make_node(
            "DequantizeLinear", node.input[:3], [node.name + "_in_dequant1"], node.name + "_in_dequant1"
        )

        in_dq2 = onnx.helper.make_node(
            "DequantizeLinear", node.input[3:6], [node.name + "_in_dequant2"], node.name + "_in_dequant2"
        )

        # update scale initializer
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, node.input[6] + "_scale")
        inits.extend([bias_scale_initializer])

        # update zero initializer
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, node.input[6] + "_zero_point")
        inits.extend([bias_zp_initializer])
        in_dq3 = onnx.helper.make_node(
            "DequantizeLinear",
            [node.input[8], bias_scale_initializer.name, bias_zp_initializer.name],
            [node.name + "_in_dequant3"],
        )

        inputs = [in_dq1.name, in_dq2.name, in_dq3.name]
        add_nodes.extend([in_dq1, in_dq2, in_dq3])

        # output q
        out_q = onnx.helper.make_node(
            "QuantizeLinear", [node.name + "_out", node.input[6], node.input[7]], node.output, node.name + "_out_quant"
        )
        outputs = [node.name + "_out"]
        add_nodes.append(out_q)

        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(attribute_to_kwarg(attribute))

        gemm_node = onnx.helper.make_node("Gemm", inputs, outputs, node.name + "_convert", **kwargs)
        add_nodes.append(gemm_node)
        return True, add_nodes, inits

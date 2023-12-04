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
"""LSTM Operator."""

import numpy
import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, op_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, ms_domain


@op_registry(op_types="LSTM")
class LSTMOperator(Operator):
    """LSTM Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(LSTMOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Do quantizaion."""
        return

    def convert_check(self, convert_format):
        """Check if conversion can be done."""
        node = self.node
        assert convert_format in ["dynamic"], "convert format for {} should be in ['dynamic']".format(node.op_type)

        if not self.quantizer.is_valid_quantize_weight(node.input[1]) or not self.quantizer.is_valid_quantize_weight(
            node.input[2]
        ):  # pragma: no cover
            return False

        model = self.quantizer.model
        W = model.get_initializer(node.input[1])
        R = model.get_initializer(node.input[2])

        if len(W.dims) != 3 or len(R.dims) != 3:  # pragma: no cover
            return False

        return True

    def convert(self, convert_format):
        """Convert to QOperator format."""
        node = self.node

        model = self.quantizer.model
        W = model.get_initializer(self.node.input[1])
        R = model.get_initializer(self.node.input[2])

        [W_num_dir, W_4_hidden_size, W_input_size] = W.dims
        [R_num_dir, R_4_hidden_size, R_hidden_size] = R.dims

        if self.per_channel:  # pragma: no cover
            del W.dims[0]
            del R.dims[0]
            W.dims[0] = W_num_dir * W_4_hidden_size
            R.dims[0] = R_num_dir * R_4_hidden_size

        quant_input_weight_tuple = self.quantizer.quantize_weight_per_channel(
            node.input[1], self.weight_dtype, self.weight_scheme, 0
        )
        quant_recurrent_weight_tuple = self.quantizer.quantize_weight_per_channel(
            node.input[2], self.weight_dtype, self.weight_scheme, 0
        )

        W_quant_weight = model.get_initializer(quant_input_weight_tuple[0])
        R_quant_weight = model.get_initializer(quant_recurrent_weight_tuple[0])

        W_quant_array = onnx.numpy_helper.to_array(W_quant_weight)
        R_quant_array = onnx.numpy_helper.to_array(R_quant_weight)

        W_quant_array = numpy.reshape(W_quant_array, (W_num_dir, W_4_hidden_size, W_input_size))
        R_quant_array = numpy.reshape(R_quant_array, (R_num_dir, R_4_hidden_size, R_hidden_size))

        W_quant_array = numpy.transpose(W_quant_array, (0, 2, 1))
        R_quant_array = numpy.transpose(R_quant_array, (0, 2, 1))

        W_quant_tranposed = onnx.numpy_helper.from_array(W_quant_array, quant_input_weight_tuple[0])
        R_quant_tranposed = onnx.numpy_helper.from_array(R_quant_array, quant_recurrent_weight_tuple[0])

        model.remove_initializers([W_quant_weight, R_quant_weight])
        model.add_initializer(W_quant_tranposed)
        model.add_initializer(R_quant_tranposed)

        W_quant_zp = model.get_initializer(quant_input_weight_tuple[1])
        R_quant_zp = model.get_initializer(quant_recurrent_weight_tuple[1])
        W_quant_scale = model.get_initializer(quant_input_weight_tuple[2])
        R_quant_scale = model.get_initializer(quant_recurrent_weight_tuple[2])

        if self.per_channel:  # pragma: no cover
            W_quant_zp.dims[:] = [W_num_dir, W_4_hidden_size]
            R_quant_zp.dims[:] = [R_num_dir, R_4_hidden_size]
            W_quant_scale.dims[:] = [W_num_dir, W_4_hidden_size]
            R_quant_scale.dims[:] = [R_num_dir, R_4_hidden_size]

        inputs = []
        input_len = len(node.input)
        inputs.extend([node.input[0]])
        inputs.extend([quant_input_weight_tuple[0], quant_recurrent_weight_tuple[0]])
        inputs.extend([node.input[3] if input_len > 3 else ""])
        inputs.extend([node.input[4] if input_len > 4 else ""])
        inputs.extend([node.input[5] if input_len > 5 else ""])
        inputs.extend([node.input[6] if input_len > 6 else ""])
        inputs.extend([node.input[7] if input_len > 7 else ""])
        inputs.extend(
            [
                quant_input_weight_tuple[2],
                quant_input_weight_tuple[1],
                quant_recurrent_weight_tuple[2],
                quant_recurrent_weight_tuple[1],
            ]
        )

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name == "layout":
                continue
            kwarg = attribute_to_kwarg(attribute)
            kwargs.update(kwarg)

        quant_lstm_name = node.name + "_quant"
        quant_lstm_node = onnx.helper.make_node(
            "DynamicQuantizeLSTM", inputs, node.output, quant_lstm_name, domain="com.microsoft", **kwargs
        )
        self.quantizer.remove_nodes.append(node)
        self.quantizer.new_nodes.append(quant_lstm_node)

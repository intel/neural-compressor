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

import onnx
from .base_operator import QuantOperatorBase
from onnxruntime.quantization.quant_utils import attribute_to_kwarg, ms_domain, \
                                                 QuantizedValueType
from neural_compressor.adaptor.ox_utils.util import QuantizedValue                                                 
class QGlobalAveragePool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def convert(self):
        node = self.node
        assert (node.op_type == "GlobalAveragePool")

        if len(self.quantizer.model.get_children(node)) == 0:
            return
        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        kwargs["channels_last"] = 0

        inputs = parent.input
        inputs.extend(child.input[1:])

        qnode = onnx.helper.make_node(
            "QLinear" + node.op_type,
            inputs,
            child.output,
            node.name, **kwargs)
        self.quantizer.new_nodes += [qnode]
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(parent)
        self.quantizer.remove_nodes.append(node)

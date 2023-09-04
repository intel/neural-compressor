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
"""Normalization Operator."""

import onnx

from neural_compressor.adaptor.ox_utils.operators.ops import Operator, QOperator, op_registry, qop_registry
from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg, ms_domain


@op_registry(op_types="BatchNormalization")
class BatchNormalizationOperator(Operator):
    """BatchNormalization Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(BatchNormalizationOperator, self).__init__(onnx_quantizer, onnx_node)

    def cast(self):
        """Cast node."""
        if self.dtype == "bf16":
            self.quantizer.cast_inputs(self.node, self.dtype, [0])
        else:
            self.quantizer.cast_inputs(self.node, self.dtype)
        self.quantizer.cast_outputs(self.node, self.dtype)


@op_registry(op_types="LayerNormalization")
class NormalizationOperator(Operator):
    """Normalization Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(NormalizationOperator, self).__init__(onnx_quantizer, onnx_node)

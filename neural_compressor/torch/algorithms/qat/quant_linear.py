#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Copyright (c) 2025 Intel Corporation
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
"""Quantized Linear."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_quantizer import TensorQuantizer


class QuantLinear(nn.Module):
    """Quantized version of nn.Linear."""

    def forward(self, input: torch.Tensor):
        """Add weight/input/output of quantization for the original forward method."""
        qw = self.weight_quantizer(self.weight)
        qi = self.input_quantizer(input)
        out = F.linear(qi, qw, self.bias)
        out = self.output_quantizer(out)
        return out

    def _setup(self, quant_cfg):
        """Init quantizer."""
        self.weight_quantizer = TensorQuantizer(
            data_type=quant_cfg.data_type,
            block_size=quant_cfg.group_size,
            bits=quant_cfg.bits,
            sym=quant_cfg.sym,
            if_quant=True,
            learn_exponent=False,
        )
        self.input_quantizer = TensorQuantizer(
            data_type=quant_cfg.act_data_type,
            block_size=quant_cfg.act_group_size,
            bits=quant_cfg.act_bits,
            sym=quant_cfg.act_sym,
            if_quant=True,
            learn_exponent=False,
        )
        self.output_quantizer = TensorQuantizer(
            data_type=quant_cfg.act_data_type,
            block_size=quant_cfg.act_group_size,
            bits=quant_cfg.act_bits,
            sym=quant_cfg.act_sym,
            if_quant=False,
        )
        # Currently don't quant output
        self.output_quantizer.disable()

        # TODO: remove
        self.original_weight_dtype = None if self.weight is None else self.weight.dtype

    def extra_repr(self) -> str:
        """Generate extra_repr making sure import keys exist in self.__dict__."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def __repr__(self):
        """Overriding the __repr__ method, makes the output more concise and meaningful."""
        return (
            f"QuantLinear(\n"
            f"  in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}\n"
            f"  (input_quantizer): {self.input_quantizer}\n"
            f"  (output_quantizer): {self.output_quantizer}\n"
            f"  (weight_quantizer): {self.weight_quantizer}\n"
            f")"
        )

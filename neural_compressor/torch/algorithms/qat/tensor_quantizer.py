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
"""TensorQuantizer Module."""

import torch
from torch import nn

try:
    from auto_round.data_type import get_quant_func
except ImportError:
    get_quant_func = None


class TensorQuantizer(nn.Module):
    """Tensor quantizer module."""

    def __init__(
        self,
        data_type="mx_fp8",
        bits=8,
        block_size=32,
        sym=True,
        if_quant=True,
        learn_exponent=False,
        amax=None,
        scale_shape=None,
        device=None,
    ):
        """Initialize quantizer and set up required variables."""
        super().__init__()
        self.amax = amax
        self.data_type = data_type
        self.num_bits = bits
        self.block_size = block_size
        self.sym = sym
        self._if_quant = if_quant
        self.learn_exponent = learn_exponent
        self._dequantize = False
        self._input_dtype = None
        self._fake_quant = True

        # enable quantizer
        self.enable()

        assert (
            get_quant_func is not None
        ), "The quantization function is imported from AutoRound, please install it. 'pip install auto-round'"

        # self.data_type will be overided 'mx_fp' -> 'mx_fp8'
        self.quant_func, self.data_type = get_quant_func(self.data_type, self.num_bits, self.sym)

        if scale_shape is not None:
            # E8M0 scales (exponent)
            self.register_buffer(
                "scale",
                torch.empty(scale_shape[0], scale_shape[1] // self.block_size, dtype=torch.uint8, device=device),
            )
            self.save_scale = True
        else:
            self.save_scale = False

    def forward(self, inputs: torch.Tensor):
        """Apply tensor_quant function to inputs.

        Args:
            inputs: A Tensor of type float32/float16/bfloat16.

        Returns:
            outputs: A Tensor of type output_dtype
        """

        if self._disabled or (not self._if_quant):
            self._input_dtype = inputs.dtype
            return inputs

        x = inputs
        if not x.is_contiguous():
            x = x.contiguous()

        if self.fake_quant:
            q = self._fake_quantize(x)[0]
        else:
            # TODO: add implementation
            q = self._real_quantize(x)

        return q.to(inputs.dtype)

    def _fake_quantize(self, inputs: torch.Tensor):
        """Fake quantization."""

        # the shared_exp can be trainable
        if self.learn_exponent:
            q, shared_exp, _ = self.quant_func(
                inputs,
                bits=self.num_bits,
                group_size=self.block_size,
                data_type=self.data_type,
            )
        else:
            # wrapper no_grad, because the function includes extra trainable variables
            with torch.no_grad():
                q, shared_exp, _ = self.quant_func(
                    inputs,
                    bits=self.num_bits,
                    group_size=self.block_size,
                    data_type=self.data_type,
                )

            # simple STE, since we add no_grad in the quant function
            q = q.detach() + (inputs - inputs.detach())

        if self.save_scale:
            # TODO: PACK uint8
            self.scale.data.copy_(shared_exp.detach())

        return q, shared_exp

    def _real_quantize(self, inputs: torch.Tensor):
        raise NotImplementedError("This method hasn't be implemented.")

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    def disable(self):
        """Bypass the module."""
        self._disabled = True

    def enable(self):
        """Enable the module."""
        self._disabled = False

    def weight_pack(self, weight, scale):
        """Pack weight and scale when saving."""
        original_shape = weight.shape

        # TODO: support more quantization format
        if self.data_type == "mx_fp8":
            qweight = (weight.reshape(-1, self.block_size) / torch.exp2(scale.float()).reshape(-1, 1)).to(
                torch.float8_e4m3fn
            )

            e8m0_scale = (scale + 127).to(torch.uint8)
            return qweight.reshape(original_shape), e8m0_scale.reshape(original_shape[0], -1)

        if self.data_type == "mx_fp4":
            qweight = weight.reshape(-1, self.block_size) / torch.exp2(scale.float())

            from auto_round.export.export_to_autoround.qlinear_fp import pack_fp4_to_uint8

            qweight_packed = pack_fp4_to_uint8(qweight)

            e8m0_scale = (scale + 127).to(torch.uint8)
            return qweight_packed.reshape(original_shape[0], original_shape[1] // 2), e8m0_scale.reshape(
                original_shape[0], -1
            )

    def __repr__(self):
        if self._disabled or not self._if_quant:
            return "TensorQuantizer(disabled)"

        qformat_str = f"({self.data_type}) format"
        bits_str = f"({self.num_bits}) bit"

        if self.block_size:
            bs_str = f"block_size={self.block_size}"
        else:
            bs_str = "block_size=None"

        # amax
        amax_str = f"amax={self.amax}" if self.amax is not None else "amax=?"
        # fake / real
        mode_str = "fake" if self._fake_quant else "real"
        # sym
        sym_str = "sym" if self.sym else "asym"
        # quant enable
        qflag = "quant" if self._if_quant else "no-quant"

        return f"TensorQuantizer({qformat_str} {bits_str} {mode_str} {bs_str}, {amax_str} {qflag})"

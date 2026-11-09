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
"""Compatibility shim for the legacy QAT ``TensorQuantizer``.

The implementation moved to
:mod:`neural_compressor.torch.quantization.fake_quant.quantizer`. The old
constructor (auto_round ``data_type``/``bits`` keyword arguments) is still
accepted for one release and translated to a
:class:`~neural_compressor.torch.quantization.fake_quant.options.QuantizerConfig`.
"""

from __future__ import annotations

import warnings

from neural_compressor.torch.quantization.fake_quant.options import QuantizerConfig
from neural_compressor.torch.quantization.fake_quant.quantizer import TensorQuantizer as _TensorQuantizer

_LEGACY_DATA_TYPES = {
    "mx_fp8": "mxfp8",
    "mx_fp4": "mxfp4",
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
}


class TensorQuantizer(_TensorQuantizer):
    """Deprecated alias with the pre-refactor constructor."""

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
        """Translate the legacy arguments and warn once."""
        warnings.warn(
            "algorithms.qat.TensorQuantizer is deprecated; use " "neural_compressor.torch.quantization.TensorQuantizer",
            DeprecationWarning,
            stacklevel=2,
        )
        if learn_exponent:
            raise NotImplementedError("learn_exponent is not supported by the fake-quant quantizer")
        if device is not None:
            warnings.warn("`device` is ignored; the quantizer follows its module", stacklevel=2)
        fmt = _LEGACY_DATA_TYPES.get(data_type)
        if fmt is None:
            raise ValueError(f"unsupported legacy data_type {data_type!r}; known: {sorted(_LEGACY_DATA_TYPES)}")
        scheme = "static" if (fmt == "fp8_e4m3" and amax is not None) else "dynamic"
        super().__init__(
            QuantizerConfig(
                format=fmt,
                enabled=bool(if_quant),
                scheme=scheme,
                amax=amax,
            )
        )
        self.sym = sym
        self.block_size = block_size
        self._if_quant = bool(if_quant)
        self._fake_quant = True
        self.save_scale = scale_shape is not None
        if self.save_scale:
            self.register_buffer(
                "scale",
                torch.empty(scale_shape[0], scale_shape[1] // block_size, dtype=torch.uint8, device=device),
            )
        self.quant_func = None

    def disable(self):
        """Bypass this quantizer, preserving the legacy API."""
        self.enabled = False

    def enable(self):
        """Enable this quantizer when legacy if_quant is true."""
        self.enabled = self._if_quant

    def _fake_quantize(self, inputs: torch.Tensor):
        """Run the legacy auto-round quantizer and return output/shared exponent."""
        if self.quant_func is None:
            try:
                from auto_round.data_type import get_quant_func
            except ImportError as exc:
                raise ImportError("legacy _fake_quantize requires auto-round") from exc
            self.quant_func, self.data_type = get_quant_func(self.data_type, self.num_bits, self.sym)
        with torch.no_grad():
            q, shared_exp, _ = self.quant_func(
                inputs.contiguous(),
                bits=self.num_bits,
                group_size=self.block_size,
                data_type=self.data_type,
            )
        q = q.detach() + (inputs - inputs.detach())
        if self.save_scale:
            self.scale.data.copy_(shared_exp.detach())
        return q, shared_exp

    def _real_quantize(self, inputs: torch.Tensor):
        """Real quantization remains outside the legacy compatibility boundary."""
        raise NotImplementedError("real quantization is not implemented by the legacy QAT shim")

    def weight_pack(self, weight, scale):
        """Pack legacy MXFP8/MXFP4 weights for auto-round export."""
        original_shape = weight.shape
        if self.data_type == "mx_fp8":
            qweight = (weight.reshape(-1, self.block_size) / torch.exp2(scale.float()).reshape(-1, 1)).to(
                torch.float8_e4m3fn
            )
            e8m0_scale = (scale + 127).to(torch.uint8)
            return qweight.reshape(original_shape), e8m0_scale.reshape(original_shape[0], -1)
        if self.data_type == "mx_fp4":
            from auto_round.export.export_to_autoround.qlinear_fp import pack_fp4_to_uint8

            qweight = weight.reshape(-1, self.block_size) / torch.exp2(scale.float())
            qweight_packed = pack_fp4_to_uint8(qweight)
            e8m0_scale = (scale + 127).to(torch.uint8)
            return qweight_packed.reshape(original_shape[0], original_shape[1] // 2), e8m0_scale.reshape(
                original_shape[0], -1
            )
        raise NotImplementedError(f"weight_pack does not support {self.data_type!r}")

    def __repr__(self):
        if not self.enabled or not self._if_quant:
            return "TensorQuantizer(disabled)"
        amax_str = f"amax={self.amax}" if self.amax is not None else "amax=?"
        mode_str = "fake" if self._fake_quant else "real"
        return (
            f"TensorQuantizer(({self.data_type}) format ({self.num_bits}) bit {mode_str} "
            f"block_size={self.block_size}, {amax_str} quant)"
        )


__all__ = ["TensorQuantizer"]

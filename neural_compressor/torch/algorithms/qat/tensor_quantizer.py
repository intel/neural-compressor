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


__all__ = ["TensorQuantizer"]

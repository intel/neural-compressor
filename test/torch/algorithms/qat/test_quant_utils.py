# -*- coding: utf-8 -*-
# Copyright (c) 2025 Intel Corporation
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
"""Tests for the export helpers kept in ``algorithms.qat.quant_utils``."""

import pytest
import torch.nn as nn

from neural_compressor.torch.algorithms.qat.quant_utils import (
    get_quantization_format,
    is_quantlinear,
)
from neural_compressor.torch.quantization.fake_quant import ModuleQuantConfig, QuantizerConfig, QuantLinear


def _quant_linear(weight_format):
    config = ModuleQuantConfig(weight=QuantizerConfig(format=weight_format))
    return QuantLinear.from_float(nn.Linear(8, 4, bias=False), config)


def test_is_quantlinear():
    assert is_quantlinear(_quant_linear("mxfp8"))
    assert not is_quantlinear(nn.Linear(8, 4))


@pytest.mark.parametrize(
    "weight_format,expected",
    [("mxfp8", "MXFP8"), ("mxfp4", "MXFP4"), ("bf16", None)],
)
def test_get_quantization_format(weight_format, expected):
    assert get_quantization_format(_quant_linear(weight_format)) == expected


def test_get_quantization_format_unsupported_raises():
    with pytest.raises(NotImplementedError):
        get_quantization_format(_quant_linear("fp8_e4m3"))

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
"""Tests for QuantLinear and the legacy TensorQuantizer shim."""

import warnings

import pytest
import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.qat.quant_linear import QuantLinear
from neural_compressor.torch.algorithms.qat.tensor_quantizer import TensorQuantizer
from neural_compressor.torch.quantization.fake_quant import ModuleQuantConfig, QuantizerConfig


def build_quant_linear(in_features=32, out_features=16, bias=True):
    source = nn.Linear(in_features, out_features, bias=bias)
    config = ModuleQuantConfig(
        weight=QuantizerConfig(format="mxfp8"),
        input=QuantizerConfig(format="mxfp8"),
    )
    return QuantLinear.from_float(source, config)


@pytest.mark.parametrize("bias", [True, False])
def test_quant_linear_forward_and_backward(bias):
    torch.manual_seed(42)
    quantized = build_quant_linear(bias=bias)
    x = torch.randn(32, 3).t().requires_grad_(True)  # non-contiguous input
    out = quantized(x)
    assert out.shape == (3, 16)
    assert out.dtype == x.dtype

    out.sum().backward()
    assert quantized.weight.grad is not None
    if bias:
        assert quantized.bias.grad is not None
    else:
        assert quantized.bias is None


def test_quant_linear_state_dict_keys_are_unchanged():
    quantized = build_quant_linear()
    keys = set(quantized.state_dict())
    assert "weight" in keys
    assert "bias" in keys
    assert not any(key.startswith("_quant") for key in keys)


def test_output_quantizer_disabled_by_default():
    quantized = build_quant_linear()
    assert not quantized.output_quantizer.enabled
    assert quantized.weight_quantizer.enabled
    assert quantized.input_quantizer.enabled


def test_legacy_tensor_quantizer_kwargs():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quantizer = TensorQuantizer(data_type="mx_fp8", bits=8)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert quantizer.format.name == "mxfp8"
    assert quantizer.num_bits == 8

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp4 = TensorQuantizer(data_type="mx_fp4", bits=4)
    assert fp4.format.name == "mxfp4"


def test_legacy_tensor_quantizer_unknown_type_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            TensorQuantizer(data_type="int3")



def test_legacy_enable_disable_and_repr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantizer = TensorQuantizer(data_type="mx_fp8")
    quantizer.disable()
    assert not quantizer.enabled
    assert repr(quantizer) == "TensorQuantizer(disabled)"
    quantizer.enable()
    assert quantizer.enabled
    assert "mx_fp8" in repr(quantizer)


def test_legacy_real_quantize_has_explicit_boundary():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantizer = TensorQuantizer(data_type="mx_fp8")
    with pytest.raises(NotImplementedError):
        quantizer._real_quantize(torch.randn(2, 32))

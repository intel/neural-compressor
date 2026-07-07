#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Intel Corporation
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
"""Tests for split quantize/dequantize QDQ layer functionality.

Verifies that running layer.call_dq(layer.call_q(x)) on a single QDQLayer instance
produces the same result as the combined layer(x) path.
"""

import pytest
from jax import numpy as jnp

from neural_compressor.jax.quantization.layers_dynamic import DynamicQDQLayer
from neural_compressor.jax.quantization.layers_static import StaticQDQLayer

_fp8_dtypes = ["float8_e4m3fn", "float8_e5m2"]
_int_dtypes = ["int8"]
_all_dtypes = _fp8_dtypes + _int_dtypes


def _make_test_inputs(model_dtype):
    """Create representative test inputs."""
    return [
        jnp.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5], dtype=model_dtype).reshape(1, 8),
        jnp.linspace(-50000.0, 50000.0, 16, dtype=model_dtype).reshape(2, 8),
        jnp.array([0.01, -0.01, 0.005, -0.005, 0.001, -0.001, 0.02, -0.02], dtype=model_dtype).reshape(1, 8),
    ]


@pytest.mark.parametrize("fixed_range", [True, False], ids=["fixed_range=True", "fixed_range=False"])
@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.smoke_test_if("model_dtype=bfloat16-activation_dtype=float8_e5m2")
def test_dynamic_qdq_split_equivalence(model_dtype, activation_dtype, fixed_range):
    """Test that layer.call_dq(layer.call_q(x)) == layer(x) on a single DynamicQDQLayer."""

    test_inputs = _make_test_inputs(model_dtype)
    if fixed_range:
        fixed_range = (-3.0, 3.0)
    else:
        fixed_range = None

    layer_qdq = DynamicQDQLayer(
        name="dynamic_qdq", activation_dtype=jnp.dtype(activation_dtype), dtype=model_dtype, fixed_range=fixed_range
    )
    layer_splitted = DynamicQDQLayer(
        name="dynamic_split", activation_dtype=jnp.dtype(activation_dtype), dtype=model_dtype, fixed_range=fixed_range
    )
    layer_qdq.add_variables()
    layer_splitted.add_variables()

    for test_input in test_inputs:
        combined_output = layer_qdq(test_input)

        quantized = layer_splitted.call_q(test_input)
        split_output = layer_splitted.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Dynamic QDQ split mismatch for dtype={activation_dtype}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )

    assert quantized.dtype == jnp.dtype(
        activation_dtype
    ), f"Expected call_q output dtype {activation_dtype}, got {quantized.dtype}"

    assert split_output.dtype == model_dtype, f"Expected call_dq output dtype {model_dtype}, got {split_output.dtype}"


@pytest.mark.parametrize("fixed_range", [True, False], ids=["fixed_range=True", "fixed_range=False"])
@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize("const_scale", [False, True], ids=["const_scale=False", "const_scale=True"])
@pytest.mark.smoke_test_if("const_scale=True-model_dtype=float32-activation_dtype=float8_e4m3fn")
def test_static_qdq_split_equivalence(model_dtype, activation_dtype, const_scale, fixed_range):
    """Test that layer.call_dq(layer.call_q(x)) == layer(x) on a single StaticQDQLayer."""

    test_inputs = _make_test_inputs(model_dtype)
    if fixed_range:
        fixed_range = (-4.0, 4.0)
    else:
        fixed_range = None

    layer_qdq = StaticQDQLayer(
        name="static_qdq",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype=model_dtype,
        const_scale=const_scale,
        fixed_range=fixed_range,
    )
    layer_splitted = StaticQDQLayer(
        name="static_split",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype=model_dtype,
        const_scale=const_scale,
        fixed_range=fixed_range,
    )

    layer_qdq.add_observers()
    layer_qdq.add_variables()
    layer_splitted.add_observers()
    layer_splitted.add_variables()

    if fixed_range is None:
        # Calibration: run inputs through observers
        for test_input in test_inputs:
            layer_qdq(test_input)
            layer_splitted.call_q(test_input)

    layer_qdq.convert()
    layer_qdq.post_quantization_cleanup()
    layer_splitted.convert()
    layer_splitted.post_quantization_cleanup()

    for test_input in test_inputs:
        combined_output = layer_qdq(test_input)

        quantized = layer_splitted.call_q(test_input)
        split_output = layer_splitted.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Static QDQ split mismatch for dtype={activation_dtype}, "
            f"const_scale={const_scale}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )

    assert quantized.dtype == jnp.dtype(
        activation_dtype
    ), f"Expected call_q output dtype {activation_dtype}, got {quantized.dtype}"

    assert split_output.dtype == model_dtype, f"Expected call_dq output dtype {model_dtype}, got {split_output.dtype}"

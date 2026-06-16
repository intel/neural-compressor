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


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.smoke_test_if("model_dtype=bfloat16-activation_dtype=float8_e4m3fn")
def test_dynamic_qdq_split_equivalence(model_dtype, activation_dtype):
    """Test that layer.call_dq(layer.call_q(x)) == layer(x) on a single DynamicQDQLayer."""

    test_inputs = _make_test_inputs(model_dtype)

    layer = DynamicQDQLayer(name="dynamic_split", activation_dtype=jnp.dtype(activation_dtype), dtype=model_dtype)
    layer.add_variables()

    for test_input in test_inputs:
        combined_output = layer(test_input)

        quantized = layer.call_q(test_input)
        split_output = layer.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Dynamic QDQ split mismatch for dtype={activation_dtype}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
def test_dynamic_qdq_split_fixed_range(model_dtype, activation_dtype):
    """Test split QDQ equivalence with fixed_range for dynamic layers."""
    fixed_range = (-3.0, 3.0)
    test_inputs = _make_test_inputs(model_dtype)

    layer = DynamicQDQLayer(
        name="dynamic_split_fixed",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype=model_dtype,
        fixed_range=fixed_range,
    )
    layer.add_variables()

    for test_input in test_inputs:
        combined_output = layer(test_input)

        quantized = layer.call_q(test_input)
        split_output = layer.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Dynamic QDQ split (fixed_range) mismatch for dtype={activation_dtype}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
def test_dynamic_call_q_output_dtype(activation_dtype):
    """Test that call_q outputs quantized dtype."""
    test_input = jnp.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5], dtype="float32").reshape(1, 8)

    layer = DynamicQDQLayer(name="q_dtype_check", activation_dtype=jnp.dtype(activation_dtype), dtype="float32")
    layer.add_variables()

    output = layer.call_q(test_input)
    assert output.dtype == jnp.dtype(
        activation_dtype
    ), f"Expected call_q output dtype {activation_dtype}, got {output.dtype}"


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
def test_dynamic_call_dq_output_dtype(activation_dtype):
    """Test that call_dq outputs compute dtype (float32)."""
    test_input = jnp.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5], dtype="float32").reshape(1, 8)

    layer = DynamicQDQLayer(name="dq_dtype_check", activation_dtype=jnp.dtype(activation_dtype), dtype="float32")
    layer.add_variables()

    quantized = layer.call_q(test_input)
    output = layer.call_dq(quantized)

    assert output.dtype == jnp.float32, f"Expected call_dq output dtype float32, got {output.dtype}"


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize("const_scale", [False, True], ids=["const_scale=False", "const_scale=True"])
def test_static_single_instance_split_call(model_dtype, activation_dtype, const_scale):
    """Test that layer.call_dq(layer.call_q(x)) == layer(x) on a single StaticQDQLayer instance."""
    fixed_range = (-4.0, 4.0)
    test_inputs = _make_test_inputs(model_dtype)

    layer = StaticQDQLayer(
        name="single_instance_split",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype=model_dtype,
        const_scale=const_scale,
        fixed_range=fixed_range,
    )
    layer.add_observers()
    layer.add_variables()
    layer.convert()
    layer.post_quantization_cleanup()

    for test_input in test_inputs:
        # Combined QDQ path
        combined_output = layer(test_input)

        # Split path: quantize then dequantize using same instance
        quantized = layer.call_q(test_input)
        split_output = layer.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Single-instance split mismatch for dtype={activation_dtype}, "
            f"const_scale={const_scale}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize("const_scale", [False, True], ids=["const_scale=False", "const_scale=True"])
@pytest.mark.smoke_test_if("const_scale=True-model_dtype=bfloat16-activation_dtype=float8_e5m2")
def test_static_single_instance_split_with_calibration(model_dtype, activation_dtype, const_scale):
    """Test single-instance split with actual calibration (no fixed_range)."""

    test_inputs = _make_test_inputs(model_dtype)

    layer = StaticQDQLayer(
        name="split_with_calib",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype=model_dtype,
        const_scale=const_scale,
    )
    layer.add_observers()
    layer.add_variables()

    # Calibration: use call_q to observe inputs (call_dq is passthrough during calibration)
    for test_input in test_inputs:
        layer.call_q(test_input)

    layer.convert()
    layer.post_quantization_cleanup()

    for test_input in test_inputs:
        # Combined QDQ path
        combined_output = layer(test_input)

        # Split path: quantize then dequantize using same instance
        quantized = layer.call_q(test_input)
        split_output = layer.call_dq(quantized)

        assert jnp.allclose(combined_output, split_output, rtol=1e-5), (
            f"Single-instance split (calibrated) mismatch for dtype={activation_dtype}, "
            f"const_scale={const_scale}:\n"
            f"  combined: {combined_output}\n"
            f"  split:    {split_output}"
        )


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.smoke_test
def test_static_split_call_q_output_dtype(activation_dtype):
    """Test that call_q outputs the quantized dtype after post_quantization_cleanup."""
    test_input = jnp.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5], dtype="float32").reshape(1, 8)

    layer = StaticQDQLayer(
        name="split_q_dtype",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype="float32",
        const_scale=True,
        fixed_range=(-4.0, 4.0),
    )
    layer.add_observers()
    layer.add_variables()
    layer.convert()
    layer.post_quantization_cleanup()

    output = layer.call_q(test_input)
    assert output.dtype == jnp.dtype(
        activation_dtype
    ), f"Expected call_q output dtype {activation_dtype}, got {output.dtype}"


@pytest.mark.parametrize("activation_dtype", _all_dtypes, ids=[f"activation_dtype={d}" for d in _all_dtypes])
@pytest.mark.smoke_test
def test_static_split_call_dq_output_dtype(activation_dtype):
    """Test that call_dq outputs the compute dtype after post_quantization_cleanup."""
    test_input = jnp.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5], dtype="float32").reshape(1, 8)

    layer = StaticQDQLayer(
        name="split_dq_dtype",
        activation_dtype=jnp.dtype(activation_dtype),
        dtype="float32",
        const_scale=True,
        fixed_range=(-4.0, 4.0),
    )
    layer.add_observers()
    layer.add_variables()
    layer.convert()
    layer.post_quantization_cleanup()

    # First quantize, then dequantize
    quantized = layer.call_q(test_input)
    output = layer.call_dq(quantized)
    assert output.dtype == jnp.float32, f"Expected call_dq output dtype float32, got {output.dtype}"

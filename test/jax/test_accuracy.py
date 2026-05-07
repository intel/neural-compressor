#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accuracy-focused tests for JAX quantization on specific model architectures.
Tests 2-layer model with known behaviors and data patterns.

Key FP8 quantization insights:
- FP8_E5M2 has discrete representable values (not continuous scaling)
- Each quantize->dequantize (QDQ) step introduces rounding to nearest FP8 value
- Cumulative precision loss through multiple QDQ operations
- Scale factors determine the range mapping to FP8's limited precision
"""

import os

import pytest
from jax_test_utility import compute_expected_qdq_dense_output

os.environ["KERAS_BACKEND"] = "jax"
from functools import reduce

import jax
import keras
import ml_dtypes
from jax import numpy as jnp

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.utils.utility import dtype_mapping

_fp8_dtypes = ["fp8_e4m3", "fp8_e5m2"]
_int_dtypes = ["int8"]
# Valid (weight_dtype, activation_dtype) pairs:
# all fp8 cross-combinations + same-dtype integer pairs
_dtype_pairs = [(f1, f2) for f1 in _fp8_dtypes for f2 in _fp8_dtypes] + [(i, i) for i in _int_dtypes]


def _read_value(var_or_array, is_const):
    """Return numeric value whether variable has `.value` or is a plain array."""
    return var_or_array if is_const else var_or_array.value


@pytest.mark.parametrize(
    "weight_dtype,activation_dtype",
    _dtype_pairs,
    ids=[f"weight_dtype={w}-activation_dtype={a}" for w, a in _dtype_pairs],
)
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["dynamic=False", "dynamic=True"])
@pytest.mark.parametrize("c_scale", [False, True], ids=["c_scale=False", "c_scale=True"])
@pytest.mark.parametrize("c_weight", [False, True], ids=["c_weight=False", "c_weight=True"])
def test_simple_linear_model_accuracy(weight_dtype, activation_dtype, model_dtype, dynamic, c_scale, c_weight):
    """Test accuracy on a simple linear model."""

    # Build model
    model_dtype_jnp = jnp.dtype(model_dtype)
    model = keras.Sequential(
        [
            keras.Input(shape=(8,)),
            keras.layers.Dense(4, activation="linear", use_bias=False, dtype=model_dtype_jnp),
            keras.layers.Dense(1, activation="linear", use_bias=False, dtype=model_dtype_jnp),
        ]
    )

    # Set weights
    all_weights = []
    for i, layer in enumerate(model.layers):
        kernel = layer.get_weights()[0]  # These dense layers only have kernel weights
        shape = kernel.shape
        num_values = reduce(lambda x, y: x * y, shape)
        weights = jnp.linspace(-1, 1, num_values) + 0.1 * (i + 1)
        weights = weights.reshape(shape)
        layer.set_weights((weights,))
        all_weights.append(weights)

    # Prepare inputs and calibration set
    test_input = jnp.array([1.0, 2.0, 2.0, 0.0, -1.0, -3.0, 0.5, float(jnp.finfo("float8_e5m2").max + 100)]).reshape(
        1, 8
    )
    calib_tensor = jnp.arange(1, 9).reshape((1, 8))

    def calib_function(model):
        _ = model(calib_tensor)

    # Create quantization config
    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=c_scale,
            const_weight=c_weight,
        )
        q_model = quantize_model(model, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=c_scale,
            const_weight=c_weight,
        )
        q_model = quantize_model(model, config, calib_function)

    # Calculate expected outputs and scales
    expected_output, expected_activation_scales, expected_weight_scales = compute_expected_qdq_dense_output(
        test_input,
        calib_tensor,
        all_weights,
        dtype_mapping[weight_dtype],
        dtype_mapping[activation_dtype],
        dynamic=dynamic,
        model_dtype=model_dtype,
    )

    # Run quantization
    quantized_output = jax.jit(q_model)(test_input)

    # Compare results with expectations
    assert jnp.allclose(
        quantized_output, expected_output, rtol=1e-5
    ), f"Quantized output mismatch: expected {expected_output}, got {quantized_output}"

    for i, (q_layer, exp_w_scale) in enumerate(zip(q_model.layers, expected_weight_scales)):
        w_scale_val = _read_value(q_layer.w_scale, c_scale)
        assert jnp.allclose(
            jnp.array(w_scale_val), jnp.array(exp_w_scale), rtol=1e-5
        ), f"Weight scale mismatch at layer {i}: expected {exp_w_scale}, got {w_scale_val}"

    if not dynamic:
        for i, (q_layer, exp_a_scale) in enumerate(zip(q_model.layers, expected_activation_scales)):
            a_scale_val = _read_value(q_layer.a_scale, c_scale)
            assert jnp.allclose(
                jnp.array(a_scale_val), jnp.array(exp_a_scale), rtol=1e-5
            ), f"Activation scale mismatch at layer {i}: expected {exp_a_scale}, got {a_scale_val}"

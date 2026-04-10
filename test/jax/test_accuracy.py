#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accuracy-focused tests for JAX quantization on specific model architectures.
Tests various 1-layer and 2-layer models with known behaviors and data patterns.

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

dtypes_list = ["fp8_e4m3", "fp8_e5m2", "int8"]


def forward_dense(config, layer, x, calib_scale):
    weight_dtype = dtype_mapping[config.weight_dtype]
    activation_dtype = dtype_mapping[config.activation_dtype]
    orig_dtype = layer.dtype
    kernel = layer.kernel
    bias = layer.bias if layer.use_bias else None
    expected_ascale = calib_scale
    expected_wscale = jnp.max(jnp.abs(kernel)) / jnp.finfo(weight_dtype).max.astype(orig_dtype)
    qdqkernel = jax.lax.clamp(
        jnp.finfo(weight_dtype).min.astype(kernel.dtype),
        kernel / expected_wscale,
        jnp.finfo(weight_dtype).max.astype(kernel.dtype),
    ).astype(weight_dtype)
    qdqkernel = qdqkernel.astype(orig_dtype) * expected_wscale
    qdqx = jax.lax.clamp(
        jnp.finfo(activation_dtype).min.astype(x.dtype),
        x / expected_ascale,
        jnp.finfo(activation_dtype).max.astype(x.dtype),
    ).astype(activation_dtype)
    qdqx = (x / expected_ascale).astype(activation_dtype)
    qdqx = qdqx.astype(orig_dtype) * expected_ascale
    y = jnp.matmul(qdqx, qdqkernel)
    if bias is not None:
        y += bias
    y = layer.activation(y)
    return y, expected_wscale


def verify_model(model, calib_tensor, test_input):
    config = StaticQuantConfig(weight_dtype="fp8_e5m2", activation_dtype="fp8_e5m2")
    # Calculate per-layer calibration results
    calib_tensor_orig = calib_tensor
    calib_scales = []
    for layer in model.layers:
        calib_scale = jnp.max(jnp.abs(calib_tensor)) / jnp.finfo(dtype_mapping[config.activation_dtype]).max.astype(
            model.dtype
        )
        calib_scales.append(calib_scale)
        calib_tensor = layer(calib_tensor)
    calib_tensor = calib_tensor_orig
    calib_fn = lambda m: m(calib_tensor)
    q_model = quantize_model(model, config, calib_fn)

    for calib_scale, q_layer, layer in zip(calib_scales, q_model.layers, model.layers):
        x = q_layer.call(test_input)
        xexpected, expected_wscale = forward_dense(config, layer, test_input, calib_scale)
        assert jnp.allclose(
            x, xexpected, rtol=1e-5
        ), f"Output mismatch in layer {q_layer.name}: expected {xexpected}, got {x}"
        assert jnp.allclose(
            q_layer.a_scale.value, calib_scale, rtol=1e-5
        ), f"Activation scale mismatch in layer {layer.name}: expected {calib_scale}, got {q_layer.a_scale.value}"
        assert jnp.allclose(
            q_layer.w_scale.value, expected_wscale, rtol=1e-5
        ), f"Weight scale mismatch in layer {layer.name}: expected {expected_wscale}, got {q_layer.w_scale.value}"
        test_input = x  # For next layer input


@pytest.mark.parametrize("weight_dtype", dtypes_list, ids=[f"(weight_dtype={dtype})" for dtype in dtypes_list])
@pytest.mark.parametrize("activation_dtype", dtypes_list, ids=[f"(activation_dtype={dtype})" for dtype in dtypes_list])
@pytest.mark.parametrize("dynamic", [False, True], ids=["(dynamic=False)", "(dynamic=True)"])
@pytest.mark.parametrize("const_scale", [False, True], ids=["(const_scale=False)", "(const_scale=True)"])
@pytest.mark.parametrize("const_weight", [False, True], ids=["(const_weight=False)", "(const_weight=True)"])
def test_simple_linear_model_accuracy(weight_dtype, activation_dtype, dynamic, const_scale, const_weight):
    """Test accuracy on a simple linear model: y = 2x (no bias)."""

    if weight_dtype == "int8" or activation_dtype == "int8":
        if weight_dtype != activation_dtype:
            return  # Mixed quantization with floating-point and integer dtypes is not supported.

    # Create single layer linear model
    model = keras.Sequential([keras.layers.Dense(1, activation="linear", input_shape=(1,), use_bias=False)])

    calib_tensor = jnp.array([[2.0], [1.0], [3.0]])

    # Initialize and set known weights
    _ = model(jnp.array([[1.0]]))
    weights = jnp.array([[2.0]])
    model.layers[0].set_weights((weights,))  # y = 2x

    def calib_function(model):
        _ = model(calib_tensor)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
        )
        q_model = quantize_model(model, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
        )
        q_model = quantize_model(model, config, calib_function)
    test_input = jnp.array([[1.0], [2.0], [2.0], [0.0], [-1.0]])
    expected_output, expected_activation_scale, expected_weight_scale = compute_expected_qdq_dense_output(
        test_input, calib_tensor, weights, dtype_mapping[weight_dtype], dtype_mapping[activation_dtype], dynamic=dynamic
    )

    quantized_output = jax.jit(q_model)(test_input)
    print(f"quantized_output: {quantized_output}")
    print(f"expected_output: {expected_output}")
    print(f"Weight scale: {q_model.layers[0].w_scale}, Expected weight scale: {expected_weight_scale}")
    if not dynamic:
        print(f"Activation scale: {q_model.layers[0].a_scale}, Expected activation scale: {expected_activation_scale}")

    def _read_value(var_or_array):
        """Return numeric value whether variable has `.value` or is a plain array."""
        return var_or_array.value if hasattr(var_or_array, "value") else var_or_array

    assert jnp.allclose(
        quantized_output, expected_output, rtol=1e-5
    ), f"Quantized output mismatch: expected {expected_output}, got {quantized_output}"

    w_scale_val = _read_value(q_model.layers[0].w_scale)
    assert jnp.allclose(
        w_scale_val, expected_weight_scale, rtol=1e-5
    ), f"Weight scale mismatch: expected {expected_weight_scale}, got {w_scale_val}"

    if not dynamic:
        a_scale_val = _read_value(q_model.layers[0].a_scale)
        assert jnp.allclose(
            a_scale_val, expected_activation_scale, rtol=1e-5
        ), f"Activation scale mismatch: expected {expected_activation_scale}, got {a_scale_val}"


def test_simple_linear_model_with_verify_util():
    """Test accuracy on a simple linear model using verify_model utility: y = 2x (no bias)."""
    # Create single layer linear model - same as test_simple_linear_model_accuracy
    model = keras.Sequential(
        [
            keras.Input(shape=(8,)),
            keras.layers.Dense(4, activation="linear", use_bias=False),
            keras.layers.Dense(1, activation="linear", use_bias=False),
        ]
    )

    # Same calibration data
    calib_tensor = jnp.arange(1, 9).reshape((1, 8))
    # Initialize and set known weights
    for i, layer in enumerate(model.layers):
        kernel = layer.get_weights()[0]  # These dense layers only have kernel weights
        shape = kernel.shape
        num_values = reduce(lambda x, y: x * y, shape)
        new_weights = jnp.arange(0, 1, 1 / num_values) + i
        new_weights = new_weights.reshape(shape)
        layer.set_weights((new_weights,))

    # Same test input
    test_input = jnp.array([1.0, 2.0, 2.0, 0.0, -1.0, 3.0, -3.0, 0.5]).reshape(1, 8)

    # Use verify_model utility instead of manual verification
    verify_model(model, calib_tensor, test_input)

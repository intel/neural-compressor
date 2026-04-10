"""Test utilities for JAX quantization tests.

This file provides helper functions used by tests in test/jax/
"""

import jax
from jax import numpy as jnp


def compute_expected_qdq_dense_output(test_input, calib_data, weights, weight_dtype, activation_dtype, dynamic):
    """Compute expected output for a Dense layer after FP8/int8 quantize-dequantize (QDQ).

    Simulates the quantization pipeline used in tests.

    Args:
        test_input: Input tensor to quantize and multiply (dynamic quantization).
        calib_data: Calibration data used to determine activation scale (static quantization).
        weights: Weight matrix of the Dense layer.
        weight_dtype: ML dtype for weight quantization.
        activation_dtype: ML dtype for activation quantization.
        dynamic: If True, compute scales from the test input (dynamic), otherwise compute them from the calibration data (static).

    Returns:
        tuple: (expected_output, activation_scale, weight_scale)
    """
    orig_dtype = jnp.float32
    # Use iinfo for integer dtypes and finfo for floating dtypes
    if jnp.issubdtype(weight_dtype, jnp.integer):
        w_max = jnp.array(jnp.iinfo(weight_dtype).max).astype(orig_dtype)
        w_min = jnp.array(jnp.iinfo(weight_dtype).min).astype(orig_dtype)
    else:
        w_max = jnp.finfo(weight_dtype).max.astype(orig_dtype)
        w_min = jnp.finfo(weight_dtype).min.astype(orig_dtype)

    if jnp.issubdtype(activation_dtype, jnp.integer):
        a_max = jnp.array(jnp.iinfo(activation_dtype).max).astype(orig_dtype)
        a_min = jnp.array(jnp.iinfo(activation_dtype).min).astype(orig_dtype)
    else:
        a_max = jnp.finfo(activation_dtype).max.astype(orig_dtype)
        a_min = jnp.finfo(activation_dtype).min.astype(orig_dtype)

    # Compute scales
    input_samples = test_input if dynamic else calib_data
    # Activation scale: for integer dtypes use asymmetric range-based formula
    if jnp.issubdtype(activation_dtype, jnp.integer):
        int_max = jnp.array(jnp.iinfo(activation_dtype).max).astype(orig_dtype)
        int_min = jnp.array(jnp.iinfo(activation_dtype).min).astype(orig_dtype)
        input_max = jnp.max(input_samples).astype(orig_dtype)
        input_min = jnp.min(input_samples).astype(orig_dtype)
        a_scale = (input_max - input_min) / (int_max - int_min)
        # zero point is used for asymmetric quantization
        a_zero_point = jnp.round(int_min - input_min / a_scale)
    else:
        a_scale = jnp.max(jnp.abs(input_samples)) / a_max
        a_zero_point = None

    w_scale = jnp.max(jnp.abs(weights)) / w_max

    # QDQ weights (with clamp)
    qdq_weights = jax.lax.clamp(w_min, weights / w_scale, w_max).astype(weight_dtype)
    qdq_weights = qdq_weights.astype(orig_dtype) * w_scale

    # QDQ input (with clamp to activation range)
    if jnp.issubdtype(activation_dtype, jnp.integer):
        # asymmetric integer quantization: round(x/scale) + zero_point
        val = jnp.round(test_input / a_scale) + a_zero_point
        val = jnp.clip(val, a_min.astype(val.dtype), a_max.astype(val.dtype)).astype(activation_dtype)
        qdq_input = (val.astype(orig_dtype) - a_zero_point) * a_scale
    else:
        qdq_input = jax.lax.clamp(
            a_min.astype(test_input.dtype),
            test_input / a_scale,
            a_max.astype(test_input.dtype),
        ).astype(activation_dtype)
        qdq_input = qdq_input.astype(orig_dtype) * a_scale

    # Compute output
    output = jnp.matmul(qdq_input, qdq_weights)

    return output, a_scale, w_scale

"""Test utilities for JAX quantization tests.

This file provides helper functions used by tests in test/jax/
"""

import ml_dtypes
import numpy as np


def _dtype_min_max(dtype, out_dtype):
    """Return (min, max) representable values for dtype, cast to out_dtype."""
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else ml_dtypes.finfo(dtype)
    return np.array(info.min, dtype=out_dtype), np.array(info.max, dtype=out_dtype)


def compute_expected_qdq_dense_output(
    test_input, calib_data, weights, weight_dtype, activation_dtype, dynamic, model_dtype
):
    """Compute expected output for a Dense layer after FP8/int8 quantize-dequantize (QDQ).

    Simulates the quantization pipeline used in tests.

    Args:
        test_input: Input tensor to quantize and multiply (dynamic quantization).
        calib_data: Calibration data used to determine activation scale (static quantization).
        weights: Weight matrix of the Dense layer.
        weight_dtype: ML dtype for weight quantization.
        activation_dtype: ML dtype for activation quantization.
        dynamic: If True, compute scales from the test input (dynamic), otherwise compute them from the calibration data (static).
        model_dtype: Dtype of the original Keras model.

    Returns:
        tuple: (expected_output, activation_scale, weight_scale)
    """
    # Convert inputs to model_dtype numpy arrays to decouple from JAX
    test_input = np.asarray(test_input, dtype=model_dtype)
    calib_data = np.asarray(calib_data, dtype=model_dtype)
    weights = np.asarray(weights, dtype=model_dtype)

    w_dtype_min, w_dtype_max = _dtype_min_max(weight_dtype, out_dtype=model_dtype)
    a_dtype_min, a_dtype_max = _dtype_min_max(activation_dtype, out_dtype=model_dtype)

    # Compute scales
    input_samples = test_input if dynamic else calib_data
    if np.issubdtype(activation_dtype, np.integer):
        input_max = np.max(input_samples)
        input_min = np.min(input_samples)
        a_scale = np.array((input_max - input_min) / (a_dtype_max - a_dtype_min), dtype=model_dtype)
        a_zero_point = np.round(a_dtype_min - input_min / a_scale)
    else:
        a_scale = np.array(np.max(np.abs(input_samples)) / a_dtype_max, dtype=model_dtype)
        a_zero_point = None

    w_scale = np.array(np.max(np.abs(weights)) / w_dtype_max, dtype=model_dtype)

    # QDQ weights
    if np.issubdtype(weight_dtype, np.integer):
        qdq_weights = np.clip(np.round(weights / w_scale), w_dtype_min, w_dtype_max).astype(weight_dtype)
    else:
        qdq_weights = np.clip(weights / w_scale, w_dtype_min, w_dtype_max).astype(weight_dtype)
    qdq_weights = qdq_weights.astype(model_dtype) * w_scale

    # QDQ input
    if np.issubdtype(activation_dtype, np.integer):
        val = np.round(test_input / a_scale) + a_zero_point
        val = np.clip(val, a_dtype_min, a_dtype_max).astype(activation_dtype)
        qdq_input = (val.astype(model_dtype) - a_zero_point) * a_scale
    else:
        qdq_input = np.clip(test_input / a_scale, a_dtype_min, a_dtype_max).astype(activation_dtype)
        qdq_input = qdq_input.astype(model_dtype) * a_scale

    # Compute output
    output = np.matmul(qdq_input, qdq_weights)

    return output, a_scale, w_scale

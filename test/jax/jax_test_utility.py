"""Test utilities for JAX quantization tests.

This file provides helper functions used by tests in test/jax/
"""

import os

import ml_dtypes
import numpy as np
from jax import numpy as jnp
from PIL import Image


def _dtype_min_max(dtype, out_dtype):
    """Return (min, max) representable values for dtype, cast to out_dtype."""
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else ml_dtypes.finfo(dtype)
    return np.array(info.min, dtype=out_dtype), np.array(info.max, dtype=out_dtype)


def _matmul(a, b):
    """Matrix multiply matching JAX's accumulation behavior.

    JAX accumulates bfloat16 matmul in float32 and casts the result back to bfloat16.
    Replicate that here so numpy results match JAX.
    """
    if a.dtype == ml_dtypes.bfloat16:
        return np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(ml_dtypes.bfloat16)
    return np.matmul(a, b)


def compute_expected_qdq_dense_output(
    test_input, calib_data, weights, weight_dtype, activation_dtype, dynamic, model_dtype
):
    """Compute expected output for one or more Dense layers after FP8/int8 quantize-dequantize (QDQ).

    Simulates the quantization pipeline used in tests.

    Args:
        test_input: Input tensor to quantize and multiply (dynamic quantization).
        calib_data: Calibration data used to determine activation scale (static quantization).
        weights: A list of model's weight matrices.
        weight_dtype: ML dtype for weight quantization.
        activation_dtype: ML dtype for activation quantization.
        dynamic: If True, compute scales from the test input (dynamic), otherwise compute them from the calibration data (static).
        model_dtype: Dtype of the original Keras model.

    Returns:
        tuple: (expected_output, first_layer_activation_scale, first_layer_weight_scale)
    """
    w_dtype_min, w_dtype_max = _dtype_min_max(weight_dtype, out_dtype=model_dtype)
    a_dtype_min, a_dtype_max = _dtype_min_max(activation_dtype, out_dtype=model_dtype)

    # Convert inputs to model_dtype numpy arrays
    current_input = np.asarray(test_input, dtype=model_dtype)
    current_calib = np.asarray(calib_data, dtype=model_dtype)

    all_a_scales = []
    all_w_scales = []

    for layer_weights in weights:
        layer_weights = np.asarray(layer_weights, dtype=model_dtype)

        # Compute scales
        input_samples = current_input if dynamic else current_calib
        if np.issubdtype(activation_dtype, np.integer):
            input_max = np.max(input_samples)
            input_min = np.min(input_samples)
            a_scale = np.array((input_max - input_min) / (a_dtype_max - a_dtype_min), dtype=model_dtype)
            a_zero_point = np.round(a_dtype_min - input_min / a_scale)
        else:
            a_scale = np.array(np.max(np.abs(input_samples)) / a_dtype_max, dtype=model_dtype)
            a_zero_point = None

        w_scale = np.array(np.max(np.abs(layer_weights)) / w_dtype_max, dtype=model_dtype)

        all_a_scales.append(a_scale)
        all_w_scales.append(w_scale)

        # QDQ weights
        if np.issubdtype(weight_dtype, np.integer):
            qdq_weights = np.clip(np.round(layer_weights / w_scale), w_dtype_min, w_dtype_max).astype(weight_dtype)
        else:
            qdq_weights = np.clip(layer_weights / w_scale, w_dtype_min, w_dtype_max).astype(weight_dtype)
        qdq_weights = qdq_weights.astype(model_dtype) * w_scale

        # QDQ input
        if np.issubdtype(activation_dtype, np.integer):
            val = np.round(current_input / a_scale) + a_zero_point
            val = np.clip(val, a_dtype_min, a_dtype_max).astype(activation_dtype)
            qdq_input = (val.astype(model_dtype) - a_zero_point) * a_scale
        else:
            qdq_input = np.clip(current_input / a_scale, a_dtype_min, a_dtype_max).astype(activation_dtype)
            qdq_input = qdq_input.astype(model_dtype) * a_scale

        # Propagate calib through unquantized layer to get next layer's calib input
        current_calib = _matmul(current_calib, layer_weights)

        # Compute output for this layer
        current_input = _matmul(qdq_input, qdq_weights)

    return current_input, all_a_scales, all_w_scales


def load_image(image_path, target_size, normalize):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    pixels = jnp.array(img)
    if normalize:
        normalized_pixels = pixels.astype(jnp.float32) / 255.0
        return jnp.expand_dims(normalized_pixels, 0)
    else:
        return jnp.expand_dims(pixels, 0)


def load_model_from_preset(model_type, preset, dtype="float32"):
    root_models_path = os.environ.get("MODELS_PATH", "/models")
    model_path = f"{root_models_path}/{preset}"
    if os.path.exists(model_path):
        return model_type.from_preset(model_path, dtype=dtype)
    else:
        raise Exception(f"Model path does not exist: {model_path}")

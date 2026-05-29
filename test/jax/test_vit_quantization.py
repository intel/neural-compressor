#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ViT model after quantization."""

import os
import tempfile
import time

import jax
import jax.numpy as jnp
import keras
import pytest
from jax_test_utility import compute_model_hash, load_image, load_model_from_preset
from keras.applications.imagenet_utils import decode_predictions
from keras_hub.models import ViTImageClassifier

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model


@pytest.fixture(scope="module")
def colva_beach_sq():
    repo_root_path = f"{os.path.dirname(__file__)}/../.."
    image_path = f"{repo_root_path}/examples/jax/keras/vit/colva_beach_sq.jpg"
    target_size = (224, 224)
    return load_image(image_path, target_size)


@pytest.fixture(scope="module")
def random_image():
    key = jax.random.PRNGKey(0)
    img = jax.random.randint(key, shape=(1, 224, 224, 3), minval=0, maxval=256, dtype=jnp.uint8)
    return img


def classify_image(model, image, top_k=1):
    out = model.predict(image)
    labels = decode_predictions(jnp.array(out), top=top_k)[0]
    probs = jax.nn.softmax(jnp.array(out)[0])
    top_probs = [probs[i] for i in jnp.argsort(probs, descending=True)[:top_k]]
    return [(class_name, prob) for ((_, class_name, _), prob) in zip(labels, top_probs)]


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "int8"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=int8"]
)
def test_image_classification(dynamic, model_dtype, quantization_dtype, colva_beach_sq, random_image):
    vit = load_model_from_preset(ViTImageClassifier, "vit_base_patch16_224_imagenet", model_dtype)

    expected_labels = classify_image(vit, colva_beach_sq)

    def calib_fn(model):
        _ = model.predict(random_image)

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        vit_q = quantize_model(vit, config, None)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype, activation_dtype=quantization_dtype, const_scale=True, const_weight=True
        )
        vit_q = quantize_model(vit, config, calib_fn, inplace=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vit_quantized.keras")
        vit_q.save_to_preset(save_path)
        vit_q_loaded = ViTImageClassifier.from_preset(save_path, dtype=model_dtype)

    actual_labels = classify_image(vit_q_loaded, colva_beach_sq)
    assert (
        actual_labels[0][0] == expected_labels[0][0]
    ), f"Expected top-1 label '{expected_labels[0][0]}', but got '{actual_labels[0][0]}'"
    actual_prob = actual_labels[0][1]
    expected_prob = expected_labels[0][1]
    error = abs(actual_prob - expected_prob)
    assert (
        error < 0.02
    ), f"Expected top-1 probability around {expected_prob}, but got {actual_prob}, absolute error: {error}"


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
def test_inplace_false(dynamic, random_image):
    quantization_dtype = "fp8_e4m3"
    model_dtype = "bfloat16"

    def calib_fn(model):
        _ = model.predict(random_image)

    vit = load_model_from_preset(ViTImageClassifier, "vit_base_patch16_224_imagenet", model_dtype)
    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        _calib_fn = None
    else:
        config = StaticQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        _calib_fn = calib_fn

    hash_before_quantization = compute_model_hash(vit)

    # inplace=False, measure time
    jax.clear_caches()
    start = time.perf_counter()
    vit_q = quantize_model(vit, config, _calib_fn, inplace=False)
    duration_inplace_false = time.perf_counter() - start

    # Assert original model is untouched
    hash_after_quantization = compute_model_hash(vit)
    assert hash_before_quantization == hash_after_quantization, "Original model was modified despite inplace=False"

    # Assert quantized model is not original
    assert vit_q is not vit
    hash_quantized = compute_model_hash(vit_q)
    assert hash_quantized != hash_before_quantization, "Quantized model should differ from the original"

    # inplace=True, measure time
    jax.clear_caches()
    start = time.perf_counter()
    vit_q = quantize_model(vit, config, _calib_fn, inplace=True)
    duration_inplace_true = time.perf_counter() - start

    # Compare quantization performance
    duration_difference = duration_inplace_false - duration_inplace_true
    performance_hit = (duration_difference / duration_inplace_true) * 100
    print(f"performance hit: {performance_hit:.2f}%")

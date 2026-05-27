#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ViT model after quantization."""

import os

os.environ["KERAS_BACKEND"] = "jax"

import tempfile

import jax
import jax.numpy as jnp
import keras
import pytest
from jax import random
from jax_test_utility import load_image, load_model_from_preset
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
    key = random.PRNGKey(0)
    img = random.randint(key, shape=(1, 224, 224, 3), minval=0, maxval=256, dtype=jnp.uint8)
    return img


def classify_image(model, image, top_k=1):
    out = model.predict(image)
    labels = decode_predictions(jnp.array(out), top=top_k)[0]
    probs = jax.nn.softmax(jnp.array(out)[0])
    top_probs = [probs[i] for i in jnp.argsort(probs, descending=True)[:top_k]]
    return [(class_name, prob) for ((_, class_name, _), prob) in zip(labels, top_probs)]


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("c_scale", [False, True], ids=["c_scale=False", "c_scale=True"])
@pytest.mark.parametrize("c_weight", [False, True], ids=["c_weight=False", "c_weight=True"])
@pytest.mark.parametrize("save_as_preset", [False, True], ids=["save_as_preset=False", "save_as_preset=True"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype",
    ["fp8_e4m3", "fp8_e5m2", "int8"],
    ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2", "quantization_dtype=int8"],
)
@pytest.mark.CI_test_if(
    [
        "dynamic=True",
        "c_scale=False",
        "c_weight=False",
        "save_as_preset=False",
        "model_dtype=float32",
        "quantization_dtype=int8",
    ],
    [
        "dynamic=False",
        "c_scale=True",
        "c_weight=True",
        "save_as_preset=True",
        "model_dtype=bfloat16",
        "quantization_dtype=fp8_e4m3",
    ],
)
def test_image_classification(
    dynamic, c_scale, c_weight, save_as_preset, model_dtype, quantization_dtype, colva_beach_sq, random_image
):
    vit = load_model_from_preset(ViTImageClassifier, "vit_base_patch16_224_imagenet", model_dtype)

    expected_labels = classify_image(vit, colva_beach_sq)

    def calib_fn(model):
        _ = model.predict(random_image)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=c_scale,
            const_weight=c_weight,
        )
        vit_q = quantize_model(vit, config, None)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=c_scale,
            const_weight=c_weight,
        )
        vit_q = quantize_model(vit, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vit_quantized.keras")
        if save_as_preset:
            vit_q.save_to_preset(save_path)
            vit_q_loaded = ViTImageClassifier.from_preset(save_path, dtype=model_dtype)
        else:
            keras.saving.save_model(vit_q, save_path)
            vit_q_loaded = keras.saving.load_model(save_path)

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

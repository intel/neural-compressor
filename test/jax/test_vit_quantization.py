#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ViT model after quantization."""

import os

os.environ["KERAS_BACKEND"] = "jax"

import tempfile

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
    return load_image(image_path, target_size, True)


@pytest.fixture(scope="module")
def random_image():
    key = random.PRNGKey(0)
    img = random.uniform(key, shape=(1, 224, 224, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)
    return img


def classify_image(model, input, labels_n=1):
    out = model(input)
    labels = decode_predictions(jnp.array(out), top=labels_n)[0]
    return [class_name for (_, class_name, _) in labels]


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
def test_image_classification(dynamic, model_dtype, colva_beach_sq, random_image):
    expected_labels = ["seashore", "sandbar", "lakeside", "promontory", "beacon"]
    quantization_dtype = "fp8_e4m3"
    vit = load_model_from_preset(ViTImageClassifier, "vit_base_patch16_224_imagenet", model_dtype)

    def calib_fn(model):
        _ = model(random_image)

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        vit_q = quantize_model(vit, config, None)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype, activation_dtype=quantization_dtype, const_scale=True, const_weight=True
        )
        vit_q = quantize_model(vit, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vit_quantized.keras")
        keras.saving.save_model(vit_q, save_path)
        vit_q = keras.saving.load_model(save_path)

    actual_labels = classify_image(vit_q, colva_beach_sq, len(expected_labels))
    assert expected_labels == actual_labels

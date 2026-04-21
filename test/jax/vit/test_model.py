#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ViT model after quantization."""

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import pytest
from jax import nn, random
from keras.applications.imagenet_utils import decode_predictions
from keras_hub.models import ViTImageClassifier
from PIL import Image

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model


@pytest.fixture(scope="module")
def colva_beach_sq():
    repo_root_path = f"{os.path.dirname(__file__)}/../../.."
    image_path = f"{repo_root_path}/examples/jax/keras/vit/colva_beach_sq.jpg"
    target_size = (224, 224)

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    normalized_pixels = jnp.array(img).astype(jnp.float32) / 255.0
    normalized_pixels = jnp.expand_dims(normalized_pixels, 0)
    return normalized_pixels


@pytest.fixture(scope="module")
def random_image():
    key = random.PRNGKey(0)
    img = random.uniform(key, shape=(1, 224, 224, 3), minval=0.0, maxval=1.0, dtype=jnp.float32)
    return img


def load_model_from_preset(model_type, preset, dtype="float32"):
    datasets_path = os.environ.get("DATASETS_PATH")
    if datasets_path is None:
        datasets_path = "/models/"

    model_path = f"{datasets_path}/{preset}"
    if os.path.exists(model_path):
        return model_type.from_preset(model_path, dtype=dtype)
    else:
        raise Exception(f"Model path does not exist: {model_path}")


def classify_image(model, input, labels_n=1):
    out = model(input)
    probs = nn.softmax(out, axis=-1)
    labels = decode_predictions(jnp.array(probs), top=labels_n)[0]
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
        config = StaticQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        vit_q = quantize_model(vit, config, calib_fn)

    actual_labels = classify_image(vit_q, colva_beach_sq, len(expected_labels))
    assert expected_labels == actual_labels

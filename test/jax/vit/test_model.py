#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ViT model after quantization."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import jax.numpy as jnp
from jax import nn
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from keras_hub.models import ViTImageClassifier
from neural_compressor.jax import quantize_model, DynamicQuantConfig, StaticQuantConfig

def load_model_from_preset(modelType, preset, allow_download=False):
    datasets_path = "/models/"
    model_path = datasets_path + preset

    if os.path.exists(model_path):
        return modelType.from_preset(model_path)
    elif allow_download:
        model = modelType.from_preset(preset)
        model.save_to_preset(model_path)
        return model
    else:
        raise Exception(f"Model path does not exist: {model_path}")

def load_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = jnp.array(img).astype(jnp.float32) / 255.0
    arr = jnp.expand_dims(arr, 0)
    return arr

def classify_image(model, input, labels_n=1):
    out = model(input)
    probs = nn.softmax(out, axis=-1)
    labels = decode_predictions(jnp.array(probs), top=labels_n)[0]
    return [class_name for (_, class_name, _) in labels]

@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
def test_image_classification(dynamic):
    image_path = "./examples/jax/keras/vit/colva_beach_sq.jpg"
    expected_labels = ['seashore', 'sandbar', 'lakeside', 'promontory', 'beacon']
    quantization_dtype = "fp8_e4m3"

    vit = load_model_from_preset(ViTImageClassifier, "vit_base_patch16_224_imagenet", True)
    input = load_image(image_path)
    
    def calib_fn(model):
        _ = model(input)

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        vit_q = quantize_model(vit, config, None)
    else:
        config = StaticQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        vit_q = quantize_model(vit, config, calib_fn)
    
    actual_labels = classify_image(vit_q, input, len(expected_labels))
    assert expected_labels == actual_labels
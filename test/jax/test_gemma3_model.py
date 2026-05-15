#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Gemma model after quantization."""

import os

os.environ["KERAS_BACKEND"] = "jax"

import random
import shutil
import string
import tempfile
from pathlib import Path

import keras
import pytest
from jax_test_utility import load_image, load_model_from_preset
from keras_hub.models import Gemma3CausalLM

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model


@pytest.fixture
def quantization_dtype():
    return "fp8_e4m3"


@pytest.fixture(scope="module")
def colva_beach_sq():
    repo_root_path = f"{os.path.dirname(__file__)}/../.."
    image_path = f"{repo_root_path}/examples/jax/keras/vit/colva_beach_sq.jpg"
    target_size = (224, 224)
    return load_image(image_path, target_size)


@pytest.fixture(scope="module")
def random_string():
    length = 50
    random.seed(0)
    return "".join(random.choices(string.ascii_letters, k=length))


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
def test_text_prompt(random_string, quantization_dtype, dynamic):
    model_dtype = "float32"
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)

    def calib_fn(model):
        _ = model.generate(random_string, max_length=100)

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype, activation_dtype=quantization_dtype, const_scale=True, const_weight=True
        )
        gemma_q = quantize_model(gemma, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        keras.saving.save_model(gemma_q, save_path)
        gemma_q = keras.saving.load_model(save_path)

    answer = gemma_q.generate("Answer what is the capital city of England. ", max_length=20, strip_prompt=True)
    print("Gemma answer: ", {answer})
    assert "The capital city of England is London" in answer


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
def test_image_recognition(colva_beach_sq, quantization_dtype, dynamic):
    model_dtype = "bfloat16"
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_4b-v1", model_dtype)

    def calib_fn(model):
        _ = model.generate(
            {
                "images": colva_beach_sq,
                "prompts": "Guess the country where this picture was taken: <start_of_image>?",
            },
            max_length=250,
        )

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype, activation_dtype=quantization_dtype, const_scale=True, const_weight=True
        )
        gemma_q = quantize_model(gemma, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(os.path.join(tmpdir, "gemma3_quantized"))
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=False)
        gemma_q.save_to_preset(save_path)
        gemma_q = Gemma3CausalLM.from_preset(str(save_path), dtype=model_dtype)

    answer = gemma_q.generate(
        {
            "images": colva_beach_sq,
            "prompts": "Enumerate all elements in the picture: <start_of_image>?",
        },
        max_length=500,
    )
    print(answer)

    elements_in_the_picture = ["beach", "chair", "tree", "building", "sea"]
    matches = sum(1 for element in elements_in_the_picture if element in answer.lower())
    assert matches >= 3, f"Expected at least 3 elements from {elements_in_the_picture} in answer (found {matches})."

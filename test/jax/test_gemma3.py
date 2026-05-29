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
import random
import string
import tempfile
import time

import jax
import keras
import pytest
from jax_test_utility import compute_model_hash, load_image, load_model_from_preset
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
@pytest.mark.parametrize("const_vars", [False, True], ids=["const_vars=False", "const_vars=True"])
@pytest.mark.parametrize("save_as_preset", [False, True], ids=["save_as_preset=False", "save_as_preset=True"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "fp8_e5m2"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2"]
)
@pytest.mark.CI_test_if(
    [
        "dynamic=True",
        "const_vars=False",
        "save_as_preset=False",
        "model_dtype=float32",
        "quantization_dtype=fp8_e5m2",
    ],
    [
        "dynamic=False",
        "const_vars=True",
        "save_as_preset=True",
        "model_dtype=bfloat16",
        "quantization_dtype=fp8_e4m3",
    ],
)
def test_text_prompt(dynamic, const_vars, save_as_preset, model_dtype, quantization_dtype, random_string):
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)

    def calib_fn(model):
        _ = model.generate(random_string, max_length=100)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config, calib_fn, inplace=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        if save_as_preset:
            gemma_q.save_to_preset(save_path)
            gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)
        else:
            keras.saving.save_model(gemma_q, save_path)
            gemma_q_loaded = keras.saving.load_model(save_path)

    answer = gemma_q_loaded.generate("Answer what is the capital city of England. ", max_length=20, strip_prompt=True)
    print("Gemma answer: ", {answer})
    assert "London" in answer


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("const_vars", [False, True], ids=["const_vars=False", "const_vars=True"])
@pytest.mark.parametrize("save_as_preset", [False, True], ids=["save_as_preset=False", "save_as_preset=True"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "fp8_e5m2"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2"]
)
@pytest.mark.CI_test_if(
    [
        "dynamic=True",
        "const_vars=True",
        "save_as_preset=True",
        "model_dtype=bfloat16",
        "quantization_dtype=fp8_e4m3",
    ],
    [
        "dynamic=False",
        "const_vars=False",
        "save_as_preset=False",
        "model_dtype=float32",
        "quantization_dtype=fp8_e5m2",
    ],
)
def test_image_recognition(dynamic, const_vars, save_as_preset, model_dtype, quantization_dtype, colva_beach_sq):
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
        config = DynamicQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        if save_as_preset:
            gemma_q.save_to_preset(save_path)
            gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)
        else:
            keras.saving.save_model(gemma_q, save_path)
            gemma_q_loaded = keras.saving.load_model(save_path)

    answer = gemma_q_loaded.generate(
        {
            "images": colva_beach_sq,
            "prompts": "Enumerate all elements in the picture: <start_of_image>?",
        },
        max_length=400,
    )
    print(answer)

    elements_in_the_picture = ["beach", "chair", "tree", "building", "sea"]
    matches = sum(1 for element in elements_in_the_picture if element in answer.lower())
    assert matches >= 3, f"Expected at least 3 elements from {elements_in_the_picture} in answer (found {matches})."


@pytest.mark.parametrize("const_vars", [False, True], ids=["const_vars=False", "const_vars=True"])
@pytest.mark.parametrize("save_as_preset", [False, True], ids=["save_as_preset=False", "save_as_preset=True"])
@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"], ids=["model_dtype=float32", "model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "fp8_e5m2"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2"]
)
@pytest.mark.CI_test_if(
    "const_vars=False", "save_as_preset=True", "model_dtype=bfloat16", "quantization_dtype=fp8_e4m3"
)
def test_static_quantization_with_incomplete_calibration(
    const_vars, save_as_preset, model_dtype, quantization_dtype, random_string, colva_beach_sq
):
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_4b-v1", model_dtype)

    # Run calibration without image in input, so vision layer won't activate during calibration
    def calib_text_fn(model):
        _ = model.generate(random_string, max_length=100)

    config = StaticQuantConfig(
        weight_dtype=quantization_dtype,
        activation_dtype=quantization_dtype,
        const_scale=const_vars,
        const_weight=const_vars,
    )
    gemma_q = quantize_model(gemma, config, calib_text_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        if save_as_preset:
            gemma_q.save_to_preset(save_path)
            gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)
        else:
            keras.saving.save_model(gemma_q, save_path)
            gemma_q_loaded = keras.saving.load_model(save_path)

    answer = gemma_q_loaded.generate(
        {
            "images": colva_beach_sq,
            "prompts": "Enumerate all elements in the picture: <start_of_image>?",
        },
        max_length=400,
        strip_prompt=True,
    )
    print(f"Gemma answer: {answer}")
    assert len(answer) > 0

    elements_in_the_picture = ["beach", "chair", "tree", "building", "sea"]
    matches = sum(1 for element in elements_in_the_picture if element in answer.lower())
    assert matches >= 3, f"Expected at least 3 elements from {elements_in_the_picture} in answer (found {matches})."


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
def test_inplace_false(dynamic, random_string):
    quantization_dtype = "fp8_e4m3"
    model_dtype = "bfloat16"
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)

    def calib_fn(model):
        _ = model.generate(random_string, max_length=100)

    if dynamic:
        config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        _calib_fn = None
    else:
        config = StaticQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)
        _calib_fn = calib_fn

    hash_before_quantization = compute_model_hash(gemma)

    # inplace=False, measure time
    jax.clear_caches()
    start = time.perf_counter()
    gemma_q = quantize_model(gemma, config, _calib_fn, inplace=False)
    duration_inplace_false = time.perf_counter() - start

    # Assert original model is untouched
    hash_after_quantization = compute_model_hash(gemma)
    assert hash_before_quantization == hash_after_quantization, "Original model was modified despite inplace=False"

    # Assert quantized model is not original
    assert gemma_q is not gemma
    hash_quantized = compute_model_hash(gemma_q)
    assert hash_quantized != hash_before_quantization, "Quantized model should differ from the original"

    # inplace=True, measure time
    jax.clear_caches()
    start = time.perf_counter()
    gemma_q = quantize_model(gemma, config, _calib_fn, inplace=True)
    duration_inplace_true = time.perf_counter() - start

    # Compare quantization performance
    duration_difference = duration_inplace_false - duration_inplace_true
    performance_hit = (duration_difference / duration_inplace_true) * 100
    print(f"performance hit: {performance_hit:.2f}%")

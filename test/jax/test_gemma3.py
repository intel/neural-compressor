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
import numpy as np
import pytest
from jax_test_utility import compute_model_hash, load_image, load_model_from_preset
from keras_hub.models import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.quantization import layers_static
from neural_compressor.jax.quantization.layers_static import QStaticCachedGemma3Attention
from neural_compressor.jax.utils.utility import dtype_mapping


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
@pytest.mark.parametrize("model_dtype", ["float32"], ids=["model_dtype=float32"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "fp8_e5m2"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2"]
)
@pytest.mark.smoke_test_if(
    "quantization_dtype=fp8_e4m3-model_dtype=float32-const_vars=False-dynamic=True",
    "quantization_dtype=fp8_e4m3-model_dtype=float32-const_vars=True-dynamic=False",
)
def test_text_prompt(dynamic, const_vars, model_dtype, quantization_dtype, random_string):
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)

    def calib_fn(model):
        _ = model.generate(random_string, max_length=100)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            weight_scale_granularity="per_channel",
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            weight_scale_granularity="per_channel",
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config, calib_fn, inplace=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        gemma_q.save_to_preset(save_path)
        gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)

    answer = gemma_q_loaded.generate("Answer what is the capital city of England.", max_length=25, strip_prompt=True)
    print("Gemma answer: ", {answer})
    assert "London" in answer


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("model_dtype", ["float32"], ids=["model_dtype=float32"])
@pytest.mark.smoke_test_if(
    "model_dtype=float32-dynamic=True",
    "model_dtype=float32-dynamic=False",
)
def test_text_prompt_dot_product_attention(dynamic, model_dtype, quantization_dtype, random_string):
    """Validate the fused dot-product-attention path (dot_product_attention_enable=True)."""
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)

    def calib_fn(model):
        _ = model.generate(random_string, max_length=100)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            weight_scale_granularity="per_channel",
            dot_product_attention_enable=True,
        )
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            weight_scale_granularity="per_channel",
            dot_product_attention_enable=True,
        )
        gemma_q = quantize_model(gemma, config, calib_fn, inplace=False)

    assert config.dot_product_attention_enable is True
    # The flag must be propagated onto the converted attention layers.
    attention_layers = [
        layer for layer in gemma_q._flatten_layers(recursive=True) if hasattr(layer, "dot_product_attention_enable")
    ]
    assert attention_layers, "No quantized attention layer exposed dot_product_attention_enable"
    assert all(layer.dot_product_attention_enable is True for layer in attention_layers)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        gemma_q.save_to_preset(save_path)
        gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)

    answer = gemma_q_loaded.generate("Answer what is the capital city of England.", max_length=25, strip_prompt=True)
    print("Gemma answer: ", {answer})
    assert "London" in answer


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic=True", "dynamic=False"])
@pytest.mark.parametrize("const_vars", [False, True], ids=["const_vars=False", "const_vars=True"])
@pytest.mark.parametrize("model_dtype", ["bfloat16"], ids=["model_dtype=bfloat16"])
@pytest.mark.parametrize(
    "quantization_dtype", ["fp8_e4m3", "fp8_e5m2"], ids=["quantization_dtype=fp8_e4m3", "quantization_dtype=fp8_e5m2"]
)
@pytest.mark.smoke_test_if(
    "quantization_dtype=fp8_e4m3-model_dtype=bfloat16-const_vars=True-dynamic=True",
    "quantization_dtype=fp8_e5m2-model_dtype=bfloat16-const_vars=False-dynamic=False",
)
def test_image_recognition(dynamic, const_vars, model_dtype, quantization_dtype, colva_beach_sq):
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
            weight_scale_granularity="per_tensor",
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config)
    else:
        config = StaticQuantConfig(
            weight_dtype=quantization_dtype,
            activation_dtype=quantization_dtype,
            weight_scale_granularity="per_tensor",
            const_scale=const_vars,
            const_weight=const_vars,
        )
        gemma_q = quantize_model(gemma, config, calib_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
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


@pytest.mark.smoke_test
def test_static_quantization_with_incomplete_calibration(random_string, colva_beach_sq):
    quantization_dtype = "fp8_e4m3"
    model_dtype = "bfloat16"
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_4b-v1", model_dtype)

    # Run calibration without image in input, so vision layer won't activate during calibration
    def calib_text_fn(model):
        _ = model.generate(random_string, max_length=100)

    config = StaticQuantConfig(
        weight_dtype=quantization_dtype,
        activation_dtype=quantization_dtype,
        const_scale=False,
        const_weight=False,
    )
    gemma_q = quantize_model(gemma, config, calib_text_fn)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
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


@pytest.mark.smoke_test
def test_inplace_false():
    quantization_dtype = "fp8_e4m3"
    model_dtype = "bfloat16"
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)
    config = DynamicQuantConfig(weight_dtype=quantization_dtype, activation_dtype=quantization_dtype)

    hash_before_quantization = compute_model_hash(gemma)

    # inplace=False, measure time
    jax.clear_caches()
    start = time.perf_counter()
    gemma_q = quantize_model(gemma, config, None, inplace=False)
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
    gemma_q = quantize_model(gemma, config, None, inplace=True)
    duration_inplace_true = time.perf_counter() - start

    # Compare quantization performance
    duration_difference = duration_inplace_false - duration_inplace_true
    performance_hit = (duration_difference / duration_inplace_true) * 100
    print(f"performance hit: {performance_hit:.2f}%")


@pytest.mark.smoke_test
def test_cached_gemma3_attention_fused_matches_fallback(monkeypatch):
    """The fused dot-product-attention path matches the einsum fallback.

    The static layer's QDQ helpers are left in passthrough (uncalibrated) mode, so
    both paths run in full precision and only the math path differs. A spy confirms
    the fused op is used for one path and not the other, guarding against silently
    falling back on both runs.
    """
    head_dim, num_query_heads, num_key_value_heads, hidden_dim = 4, 2, 1, 8
    q_dtype = dtype_mapping["fp8_e4m3"]

    layer = CachedGemma3Attention(
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
    )
    layer.build((None, None, hidden_dim))
    q_layer = QStaticCachedGemma3Attention.prepare(
        layer,
        q_dtype,
        q_dtype,
        False,
        False,
        "per_tensor",
        dot_product_attention_enable=True,
    )
    q_layer.add_observers()
    q_layer.add_variables()

    real_dot_product_attention = layers_static.ops.dot_product_attention
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_dot_product_attention(*args, **kwargs)

    monkeypatch.setattr(layers_static.ops, "dot_product_attention", spy)

    rng = np.random.default_rng(0)
    q = keras.ops.convert_to_tensor(rng.standard_normal((1, 3, num_query_heads, head_dim)).astype("float32"))
    k = keras.ops.convert_to_tensor(rng.standard_normal((1, 3, num_key_value_heads, head_dim)).astype("float32"))
    v = keras.ops.convert_to_tensor(rng.standard_normal((1, 3, num_key_value_heads, head_dim)).astype("float32"))
    attention_mask = keras.ops.ones((1, 3, 3))

    q_layer.dot_product_attention_enable = True
    fused = q_layer._compute_attention(q, k, v, attention_mask)
    assert calls["n"] == 1, "Fused path did not call ops.dot_product_attention"

    q_layer.dot_product_attention_enable = False
    fallback = q_layer._compute_attention(q, k, v, attention_mask)
    assert calls["n"] == 1, "Fallback path unexpectedly called ops.dot_product_attention"

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fused),
        keras.ops.convert_to_numpy(fallback),
        rtol=1e-3,
        atol=1e-3,
    )

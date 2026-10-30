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
import re
import string
import tempfile
import time
import warnings
from types import SimpleNamespace

import jax
import keras
import numpy as np
import pytest
from jax import numpy as jnp
from jax_test_utility import compute_model_hash, load_image, load_model_from_preset
from keras_hub.models import Gemma3Backbone, Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.vllm.context import vllm_context_scope

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.quantization import layers_static
from neural_compressor.jax.quantization.layers_dynamic import QDynamicCachedGemma3Attention
from neural_compressor.jax.quantization.layers_static import QStaticCachedGemma3Attention
from neural_compressor.jax.utils import utility
from neural_compressor.jax.utils.utility import dtype_mapping, select_kv_cache_dtype


def _record_kv_cache_dtypes(model, monkeypatch):
    """Record per-layer (key, value) cache dtypes of caches built during the next generate() trace."""
    recorded = []
    build_cache = model._build_cache

    def spy(*args, **kwargs):
        hidden_states, cache = build_cache(*args, **kwargs)
        recorded.append([tuple(jnp.dtype(value.dtype) for value in layer_cache) for layer_cache in cache])
        return hidden_states, cache

    monkeypatch.setattr(model, "_build_cache", spy)
    # Force a fresh trace so the spy runs inside generate().
    model.generate_function = None
    return recorded


def _check_recorded_kv_cache(model, recorded, expected_dtypes):
    assert recorded, "generate() did not build a KV cache"
    for layer, layer_dtypes, expected_dtype in zip(model.backbone.transformer_layers, recorded[0], expected_dtypes):
        assert layer_dtypes == (jnp.dtype(expected_dtype),) * 2
        uses_fp8_cache = jnp.dtype(expected_dtype).itemsize == 1
        assert layer.autocast is not uses_fp8_cache
        assert layer.attention.autocast is not uses_fp8_cache
    assert len(recorded[0]) == len(model.backbone.transformer_layers) == len(expected_dtypes)


def _check_model_kv_cache(model, expected_dtypes):
    token_ids = jnp.array([[1, 2, 3, 4]], dtype="int32")
    hidden_states, cache = model._build_cache(
        token_ids=token_ids,
        img_embeddings=None,
        vision_mask=None,
        padding_mask=jnp.ones_like(token_ids, dtype="bool"),
        vision_indices=None,
    )
    assert hidden_states.shape == (1, 4, model.backbone.hidden_dim)
    assert bool(jnp.all(jnp.isfinite(hidden_states)))
    assert isinstance(cache, tuple)
    assert len(cache) == len(model.backbone.transformer_layers) == len(expected_dtypes)
    for layer, layer_cache, expected_dtype in zip(model.backbone.transformer_layers, cache, expected_dtypes):
        uses_fp8_cache = jnp.dtype(expected_dtype).itemsize == 1
        assert layer.autocast is not uses_fp8_cache
        assert layer.attention.autocast is not uses_fp8_cache
        assert isinstance(layer_cache, tuple)
        assert len(layer_cache) == 2
        for value in layer_cache:
            assert value.shape == (1, 4, model.backbone.num_key_value_heads, model.backbone.head_dim)
            assert value.dtype == jnp.dtype(expected_dtype)
            assert bool(jnp.all(jnp.isfinite(value)))
    return hidden_states, cache


@pytest.mark.smoke_test
def test_build_cache_uses_fp8_per_attention_layer():
    class Attention:
        def __init__(self, use_fp8_cache):
            self._use_fp8_cache = use_fp8_cache
            self.activation_dtype = utility.dtype_mapping["fp8_e4m3"]
            self.autocast = True

        def can_use_fp8_kv_cache(self):
            return self._use_fp8_cache

    class TransformerLayer:
        def __init__(self, use_fp8_cache):
            self.attention = Attention(use_fp8_cache)
            self.autocast = True

    class Model:
        compute_dtype = "bfloat16"

        def __init__(self):
            self.backbone = type(
                "Backbone",
                (),
                {
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "transformer_layers": [
                        TransformerLayer(use_fp8_cache=True),
                        TransformerLayer(use_fp8_cache=False),
                    ],
                },
            )()

        def _build_cache(self, *args, **kwargs):
            return None

        def call_with_cache(self, **kwargs):
            return None, "hidden_states", kwargs["cache"]

    model = Model()
    model.generate_function = lambda *args, **kwargs: "stale"
    utility.causal_lm_make_replace_build_cache(model)
    assert model._build_cache.__func__ is utility._build_cache_fp8
    assert model.generate_function is None
    # Autocast is decided at cache allocation, not at installation.
    for layer in model.backbone.transformer_layers:
        assert layer.autocast is True
        assert layer.attention.autocast is True

    hidden_states, cache = model._build_cache(
        token_ids=jnp.ones((2, 3), dtype="int32"),
        img_embeddings=None,
        vision_mask=None,
        padding_mask=None,
        vision_indices=None,
    )

    assert hidden_states == "hidden_states"
    assert isinstance(cache, tuple)
    assert len(cache) == 2
    for layer_cache, dtype in zip(cache, [utility.dtype_mapping["fp8_e4m3"], jnp.dtype("bfloat16")]):
        assert isinstance(layer_cache, tuple)
        assert len(layer_cache) == 2
        for value in layer_cache:
            assert value.shape == (2, 3, 2, 4)
            assert value.dtype == dtype
            assert bool(jnp.all(value == 0))
    assert model.backbone.transformer_layers[0].autocast is False
    assert model.backbone.transformer_layers[0].attention.autocast is False
    assert model.backbone.transformer_layers[1].autocast is True
    assert model.backbone.transformer_layers[1].attention.autocast is True


@pytest.mark.smoke_test
def test_build_cache_repeated_installation_does_not_track_backbone_layers():
    class Backbone(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.transformer_layers = [keras.layers.Layer(), keras.layers.Layer(autocast=False)]
            for layer in self.transformer_layers:
                layer.attention = keras.layers.Layer()
                layer.attention.can_use_fp8_kv_cache = lambda: True
                layer.attention.activation_dtype = utility.dtype_mapping["fp8_e4m3"]

        def call(self, inputs):
            return inputs

    class Model(keras.Model):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

        def call(self, inputs):
            return self.backbone(inputs)

        def _build_cache(self):
            return "original"

    model = Model()
    model(jnp.ones((1, 1)))
    original_layers = tuple(model._layers)
    original_variables = tuple(model.variables)

    utility.causal_lm_make_replace_build_cache(model)
    utility.causal_lm_make_replace_build_cache(model)

    assert model._tracker.locked
    assert tuple(model._layers) == original_layers
    assert tuple(model.variables) == original_variables
    assert [layer.autocast for layer in model.backbone.transformer_layers] == [True, False]

    for layer in model.backbone.transformer_layers:
        assert utility.select_kv_cache_dtype(layer, "bfloat16") == utility.dtype_mapping["fp8_e4m3"]

    assert model._tracker.locked
    assert tuple(model._layers) == original_layers
    assert tuple(model.variables) == original_variables
    assert all(not layer.autocast for layer in model.backbone.transformer_layers)
    assert all(not layer.attention.autocast for layer in model.backbone.transformer_layers)


@pytest.mark.smoke_test
@pytest.mark.parametrize("api_changed", [False, True])
def test_build_cache_verifies_gemma3_api_at_import(monkeypatch, api_changed):
    import runpy
    from unittest.mock import Mock

    if api_changed:
        monkeypatch.setattr(utility.Gemma3CausalLM, "_build_cache", lambda self, token_ids: None)
    error = Mock()
    monkeypatch.setattr(utility.logger, "error", error)
    runpy.run_path(utility.__file__)

    if api_changed:
        error.assert_called_once()
        message = error.call_args.args[0]
        assert "Gemma3CausalLM._build_cache has changed" in message
        assert "_build_cache_fp8" in message
        assert "earlier compatible version of keras_hub" in message
    else:
        error.assert_not_called()


@pytest.mark.parametrize(
    ("activation_dtype", "expected"),
    [
        (dtype_mapping["fp8"], True),
        (dtype_mapping["fp8_e4m3"], True),
        (dtype_mapping["fp8_e5m2"], True),
        (jnp.dtype("int8"), False),
        (jnp.dtype("uint8"), False),
        (jnp.dtype("bool"), False),
        (jnp.dtype("bfloat16"), False),
        (jnp.dtype("float32"), False),
    ],
)
def test_fp8_cache_requires_one_byte_floating_dtype(activation_dtype, expected):
    attention = CachedGemma3Attention(head_dim=4, num_query_heads=2, num_key_value_heads=1)
    QStaticCachedGemma3Attention.prepare(
        attention, activation_dtype, activation_dtype, False, False, "per_tensor", False
    )
    assert attention._use_fp8_cache is expected


@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("calibrated", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("reload_helpers", [False, True])
@pytest.mark.parametrize("const_scale", [False, True])
def test_fp8_cache_with_incomplete_kv_calibration(
    model_dtype, calibrated, fused, reload_helpers, const_scale, monkeypatch
):
    def prepare_attention():
        attention = CachedGemma3Attention(head_dim=4, num_query_heads=2, num_key_value_heads=1, dtype=model_dtype)
        attention.build((1, 2, 8))
        QStaticCachedGemma3Attention.prepare(
            attention, dtype_mapping["fp8_e4m3"], dtype_mapping["fp8_e4m3"], const_scale, False, "per_tensor", fused
        )
        attention.add_observers()
        attention.add_variables()
        return attention

    attention = prepare_attention()
    samples = jnp.linspace(-1, 1, 8, dtype=model_dtype).reshape((1, 2, 1, 4))
    if any(calibrated):
        attention.q_qdq(samples)
    for helper, should_calibrate in zip((attention.k_qdq, attention.v_qdq), calibrated):
        if should_calibrate:
            helper(samples)
    attention.convert()
    attention.post_quantization_cleanup()

    if reload_helpers:
        restored = prepare_attention()
        restored.post_quantization_cleanup()
        for name in ("q_qdq", "attention_softmax_qdq", "k_qdq", "v_qdq"):
            store = {}
            getattr(attention, name).save_own_variables(store)
            getattr(restored, name).load_own_variables(store)
        attention = restored

    expected_dtype = dtype_mapping["fp8_e4m3"] if all(calibrated) else jnp.dtype(model_dtype)
    assert attention.can_use_fp8_kv_cache() is all(calibrated)
    assert attention._use_fp8_cache
    assert not hasattr(attention, "k_cache")
    assert not hasattr(attention, "v_cache")
    # prepare() leaves autocast alone; it's disabled at cache allocation for FP8-cache layers only.
    assert attention.autocast is True
    decoder_layer = SimpleNamespace(attention=attention, autocast=True)
    cache_dtype = select_kv_cache_dtype(decoder_layer, attention.compute_dtype)
    assert jnp.dtype(cache_dtype) == expected_dtype
    assert attention.autocast is not all(calibrated)
    assert decoder_layer.autocast is not all(calibrated)

    original_call = CachedGemma3Attention.call
    quantized_call = QStaticCachedGemma3Attention.call
    calls = []

    def record_original_call(self, *args, **kwargs):
        calls.append("original")
        return original_call(self, *args, **kwargs)

    def record_quantized_call(self, *args, **kwargs):
        calls.append("quantized")
        return quantized_call(self, *args, **kwargs)

    monkeypatch.setattr(CachedGemma3Attention, "call", record_original_call)
    monkeypatch.setattr(QStaticCachedGemma3Attention, "call", record_quantized_call)

    inputs = jnp.linspace(-1, 1, 16, dtype=model_dtype).reshape((1, 2, 8))
    mask = jnp.ones((1, 2, 2), dtype="bool")
    cache = tuple(jnp.zeros((1, 2, 1, 4), dtype=cache_dtype) for _ in range(2))

    @jax.jit
    def cached_call(cache):
        return attention(inputs, attention_mask=mask, cache=cache)

    output, cache = cached_call(cache)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert all(value.dtype == expected_dtype for value in cache)
    output, cache = cached_call(cache)
    assert bool(jnp.all(jnp.isfinite(output)))
    uncached_output = attention(inputs, attention_mask=mask)
    assert bool(jnp.all(jnp.isfinite(uncached_output)))
    tolerance = 1e-2 if model_dtype == "bfloat16" else 1e-5
    assert jnp.allclose(output, uncached_output, rtol=tolerance, atol=tolerance)

    causal_mask = jnp.tril(mask)
    causal_output = attention(inputs, attention_mask=causal_mask)
    _, prefix_cache = attention(
        inputs[:, :1], attention_mask=mask[:, :1], cache=tuple(jnp.zeros_like(value) for value in cache)
    )
    incremental_output, incremental_cache = attention(
        inputs[:, 1:], attention_mask=mask[:, :1], cache=prefix_cache, cache_update_index=1
    )
    assert jnp.allclose(incremental_output, causal_output[:, 1:], rtol=tolerance, atol=tolerance)
    for before, after in zip(prefix_cache, incremental_cache):
        assert after.dtype == expected_dtype
        np.testing.assert_array_equal(before[:, :1], after[:, :1])

    extended_cache = tuple(jnp.concatenate([value, jnp.zeros_like(value)], axis=1) for value in cache)
    extended_mask = jnp.ones((1, 2, 4), dtype="bool")
    _, unmasked_cache = attention(inputs * 2, attention_mask=extended_mask, cache=extended_cache, cache_update_index=2)
    _, masked_cache = attention(
        inputs * 2,
        attention_mask=extended_mask,
        cache=extended_cache,
        cache_update_index=2,
        cache_update_mask=jnp.array([[True, False]]),
    )
    for before, unmasked, masked in zip(extended_cache, unmasked_cache, masked_cache):
        assert masked.dtype == expected_dtype
        np.testing.assert_array_equal(masked[:, :2], before[:, :2])
        np.testing.assert_array_equal(masked[:, 2:3], unmasked[:, 2:3])
        np.testing.assert_array_equal(masked[:, 3:], before[:, 3:])
        assert bool(jnp.any(unmasked[:, 2:3] != before[:, 2:3]))
    assert calls
    assert set(calls) == ({"quantized"} if all(calibrated) else {"quantized", "original"})


@pytest.mark.parametrize("model_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("uncalibrated_helpers", [("k_qdq",), ("v_qdq",), ("k_qdq", "v_qdq")])
def test_mixed_layer_kv_cache_save_load(model_dtype, uncalibrated_helpers, monkeypatch, tmp_path):
    backbone = Gemma3Backbone(
        vocabulary_size=16,
        image_size=16,
        num_layers=4,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=4,
        dtype=model_dtype,
    )
    model = Gemma3CausalLM(backbone=backbone, preprocessor=None)
    inputs = {
        "token_ids": jnp.array([[1, 2, 3, 4]], dtype="int32"),
        "padding_mask": jnp.ones((1, 4), dtype="bool"),
    }
    attentions = [layer.attention for layer in backbone.transformer_layers]
    config = StaticQuantConfig(
        white_list=[f"^{re.escape(attention.path)}$" for attention in attentions[:2]],
    ) + DynamicQuantConfig(white_list=[f"^{re.escape(attentions[2].path)}$"])

    def calibrate(prepared):
        # Simulate missing observations without bypassing model serialization or cache allocation.
        with monkeypatch.context() as context:
            for name in uncalibrated_helpers:
                helper = getattr(prepared.backbone.transformer_layers[1].attention, name)
                context.setattr(helper, "call", lambda inputs, **kwargs: inputs)
            prepared(inputs)

    quantized = quantize_model(model, config, calibrate)
    expected_dtypes = [dtype_mapping["fp8_e4m3"], model_dtype, model_dtype, model_dtype]

    def check_layers(model):
        layers = model.backbone.transformer_layers
        assert isinstance(layers[0].attention, QStaticCachedGemma3Attention)
        assert layers[0].attention.can_use_fp8_kv_cache()
        assert isinstance(layers[1].attention, QStaticCachedGemma3Attention)
        assert not layers[1].attention.can_use_fp8_kv_cache()
        for name in ("k_qdq", "v_qdq"):
            assert getattr(layers[1].attention, name)._is_quantized is (name not in uncalibrated_helpers)
        assert isinstance(layers[2].attention, QDynamicCachedGemma3Attention)
        assert type(layers[3].attention) is CachedGemma3Attention

    check_layers(quantized)
    original_hidden, original_cache = _check_model_kv_cache(quantized, expected_dtypes)
    save_path = tmp_path / "mixed_gemma3.keras"
    keras.saving.save_model(quantized, save_path)
    restored = keras.saving.load_model(save_path)
    check_layers(restored)
    restored_hidden, restored_cache = _check_model_kv_cache(restored, expected_dtypes)
    np.testing.assert_allclose(
        np.asarray(restored_hidden, dtype="float32"), np.asarray(original_hidden, dtype="float32"), rtol=1e-5
    )
    for original_layer, restored_layer in zip(original_cache, restored_cache):
        for original, restored in zip(original_layer, restored_layer):
            np.testing.assert_array_equal(original, restored)


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
def test_text_prompt(dynamic, const_vars, model_dtype, quantization_dtype, random_string, monkeypatch):
    gemma = load_model_from_preset(Gemma3CausalLM, "gemma3_instruct_270m", model_dtype)
    num_layers = len(gemma.backbone.transformer_layers)

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

    expected_cache_dtypes = [model_dtype if dynamic else dtype_mapping[quantization_dtype]] * num_layers

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized.keras")
        gemma_q.save_to_preset(save_path)
        gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)

    recorded_cache_dtypes = _record_kv_cache_dtypes(gemma_q_loaded, monkeypatch)
    prompt = "Answer what is the capital city of England."
    answer = gemma_q_loaded.generate(prompt, max_length=25, strip_prompt=True)
    _check_recorded_kv_cache(gemma_q_loaded, recorded_cache_dtypes, expected_cache_dtypes)
    print(f"Gemma answer for prompt '{prompt}': {answer}")
    if "London" not in answer:
        warnings.warn(f"Expected 'London' in answer, but got: '{answer}'")
        prompt = "2 + 2 ="
        print("Wrong answer, trying simpler prompt:")
        answer = gemma_q_loaded.generate(prompt, max_length=20, strip_prompt=True)
        print(f"Gemma answer for prompt '{prompt}': {answer}")
        assert "4" in answer or "four" in answer.lower()


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

    def check_dot_product_attention_enable(model, expected_value):
        attention_layers = [
            layer for layer in model._flatten_layers(recursive=True) if hasattr(layer, "dot_product_attention_enable")
        ]
        assert attention_layers, "No quantized attention layer exposed dot_product_attention_enable"
        assert all(
            layer.dot_product_attention_enable is expected_value for layer in attention_layers
        ), f"Expected dot_product_attention_enable={expected_value} for all attention layers"

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
    check_dot_product_attention_enable(gemma_q, True)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gemma3_quantized_keras")
        gemma_q.save_to_preset(save_path)
        gemma_q_loaded = Gemma3CausalLM.from_preset(save_path, dtype=model_dtype)

    check_dot_product_attention_enable(gemma_q_loaded, True)

    prompt = "Answer what is the capital city of England."
    answer = gemma_q_loaded.generate(prompt, max_length=25, strip_prompt=True)
    print(f"Gemma answer for prompt '{prompt}': {answer}")
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

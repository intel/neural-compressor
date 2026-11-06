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
"""Tests for SmoothQuant (folded into the weights) in JAX static quantization."""

import json

import keras
import numpy as np
import pytest
from keras_hub.models import Backbone, Gemma3Backbone, Gemma3CausalLM, Gemma3CausalLMPreprocessor

from neural_compressor.jax import JaxBaseConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.algorithms import smooth_quant as sq

mock_tokenizer = pytest.importorskip("keras_hub.src.tests.mocks.mock_gemma3_tokenizer")

PROMPTS = ["the quick brown fox", "the earth is round"]


def tiny_gemma3(seed=0, outliers=True):
    """Small random text-only Gemma3 (float32) with a few outlier channels in the pre-attention / pre-FFW norms."""
    keras.utils.set_random_seed(seed)
    tokenizer = mock_tokenizer.MockGemma3Tokenizer()
    preprocessor = Gemma3CausalLMPreprocessor(
        image_converter=None,
        tokenizer=tokenizer,
        sequence_length=16,
        max_images_per_prompt=0,
        num_vision_tokens_per_image=0,
    )
    backbone = Gemma3Backbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        image_size=16,
        num_layers=2,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=32,
        intermediate_dim=64,
        head_dim=8,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        vision_encoder=None,
        dtype="float32",
    )
    model = Gemma3CausalLM(backbone=backbone, preprocessor=preprocessor)
    if outliers:
        rng = np.random.default_rng(seed)
        for block in backbone.transformer_layers:
            for norm in (block.pre_attention_norm, block.pre_ffw_norm):
                g = np.asarray(norm.scale).copy()
                g[rng.choice(g.size, 2, replace=False)] = 30.0
                norm.scale.assign(g)
    return model, preprocessor


def logits(model, preprocessor):
    return np.asarray(model(preprocessor.generate_preprocess(PROMPTS)))


def generate_calibration(model):
    model.generate(PROMPTS, max_length=16)


def rel_err(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def test_groups_found_for_gemma3():
    model, _ = tiny_gemma3()
    groups = sq.find_smooth_groups(model)
    names = sorted(g.name.split("/")[-1] for g in groups)
    assert names == sorted(["qkv", "gate_up", "down", "attention_output"] * 2)
    att_out = [g for g in groups if g.name.endswith("attention_output")][0]
    assert att_out.head_repeat == 2  # 4 query heads share 2 kv heads


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
def test_smoothing_preserves_outputs(alpha):
    """Folding is exact: the smoothed float model gives the same logits, with the same variables."""
    model, preprocessor = tiny_gemma3()
    x = preprocessor.generate_preprocess(PROMPTS)
    ref = logits(model, preprocessor)
    weight_paths = [w.path for w in model.weights]
    kernel_before = np.asarray(model.backbone.transformer_layers[0].ffw_linear.kernel).copy()

    sq.apply_smooth_quant(model, lambda m: m(x), alpha=alpha)

    np.testing.assert_allclose(logits(model, preprocessor), ref, rtol=1e-4, atol=1e-4)
    assert [w.path for w in model.weights] == weight_paths  # observers removed, nothing added
    assert not np.allclose(np.asarray(model.backbone.transformer_layers[0].ffw_linear.kernel), kernel_before)
    for layer in model._flatten_layers():
        assert not hasattr(layer, "_smooth_quant_observer")
        assert "call" not in layer.__dict__  # the class call is used again


def test_smoothing_factors():
    a = np.array([4.0, 1.0, 0.0, 1e6])
    w = np.array([1.0, 4.0, 1.0, 1e-6])
    s = sq.smoothing_factors(a, w, alpha=0.5)
    np.testing.assert_allclose(s[:2], [2.0, 0.5])
    assert s[2] == 1.0  # no activation statistic -> unchanged
    assert s[3] == sq.SCALE_MAX  # clipped


def test_observers_update_inside_jitted_generate():
    """Statistics are collected through the jitted CausalLM generate used for calibration in quantize_model."""
    model, preprocessor = tiny_gemma3()
    ref = logits(model, preprocessor)
    config = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", smooth_quant=True)
    qmodel = quantize_model(model, config, generate_calibration)
    # every quantized layer got a finite activation scale (the observed layers are calibrated again after smoothing)
    for layer in qmodel._flatten_layers():
        if hasattr(layer, "a_scale") and not isinstance(getattr(layer, "a_scale"), (int, float)):
            assert np.all(np.isfinite(np.asarray(layer.a_scale))), layer.path
    assert rel_err(logits(qmodel, preprocessor), ref) < 0.1


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_smooth_quant_reduces_int8_error(granularity):
    """With outlier channels, per-tensor int8 activations lose precision; SmoothQuant reduces the output error."""
    errors = {}
    for smooth in (False, True):
        model, preprocessor = tiny_gemma3()
        ref = logits(model, preprocessor)
        config = StaticQuantConfig(
            weight_dtype="int8",
            activation_dtype="int8",
            weight_scale_granularity=granularity,
            smooth_quant=smooth,
        )
        qmodel = quantize_model(model, config, generate_calibration)
        errors[smooth] = rel_err(logits(qmodel, preprocessor), ref)
    assert errors[True] < errors[False], errors


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "int8"])
def test_same_quantized_structure(dtype):
    """SmoothQuant changes values only: same layer classes and variable shapes/dtypes as without it."""
    structures = []
    for smooth in (False, True):
        model, _ = tiny_gemma3()
        config = StaticQuantConfig(
            weight_dtype=dtype, activation_dtype=dtype, const_scale=True, const_weight=True, smooth_quant=smooth
        )
        qmodel = quantize_model(model, config, generate_calibration)
        structures.append(
            (
                [type(layer).__name__ for layer in qmodel._flatten_layers()],
                [(w.path, tuple(w.shape), str(w.dtype)) for w in qmodel.weights],
            )
        )
    assert structures[0] == structures[1]


def test_unsupported_model_is_unchanged(model, calibration_data):
    """Models without a SmoothQuant rule are quantized as before, with a warning."""
    before = [np.asarray(w).copy() for w in model.weights]
    assert sq.find_smooth_groups(model) == []
    sq.apply_smooth_quant(model, lambda m: m(calibration_data))
    for w, b in zip(model.weights, before):
        np.testing.assert_array_equal(np.asarray(w), b)
    quantize_model(model, StaticQuantConfig(smooth_quant=True), lambda m: m(calibration_data))


def test_save_load_without_smooth_quant_params(tmp_path):
    """The smoothing is baked into the weights, so the saved model config doesn't carry the SmoothQuant options (it
    loads with INC versions without them) and reloads with the same outputs."""
    model, preprocessor = tiny_gemma3()
    config = StaticQuantConfig(const_scale=True, const_weight=True, smooth_quant=True)
    qmodel = quantize_model(model, config, generate_calibration)
    inputs = preprocessor.generate_preprocess(PROMPTS)
    inputs = {k: inputs[k] for k in ("token_ids", "padding_mask")}
    expected = np.asarray(qmodel.backbone(inputs))
    # the mock tokenizer has no assets to save, so save the quantized backbone (it carries the quant config)
    qmodel.backbone.save_to_preset(str(tmp_path))
    saved = (tmp_path / "config.json").read_text()
    assert "_quant_config" in saved and "smooth_quant" not in saved
    loaded = Backbone.from_preset(str(tmp_path))
    np.testing.assert_allclose(np.asarray(loaded(inputs)), expected, rtol=1e-5, atol=1e-5)


def test_config_serialization():
    config = StaticQuantConfig(weight_scale_granularity="per_tensor", smooth_quant=True, smooth_quant_alpha=0.4)
    d = config.to_dict()
    assert d["config"]["smooth_quant"] is True and d["config"]["smooth_quant_alpha"] == 0.4
    loaded = JaxBaseConfig.from_json_string(json.dumps(d))
    assert isinstance(loaded, StaticQuantConfig)
    assert loaded == config
    assert StaticQuantConfig().smooth_quant is False


def test_config_rejects_bad_alpha():
    with pytest.raises(ValueError):
        StaticQuantConfig(smooth_quant=True, smooth_quant_alpha=1.5)

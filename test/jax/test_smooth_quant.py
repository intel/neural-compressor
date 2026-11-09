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
from jax import numpy as jnp
from keras_hub.models import Backbone, Gemma3Backbone, Gemma3CausalLM, Gemma3CausalLMPreprocessor

from neural_compressor.jax import JaxBaseConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.algorithms import smooth_quant as sq

mock_tokenizer = pytest.importorskip("keras_hub.src.tests.mocks.mock_gemma3_tokenizer")

pytestmark = pytest.mark.smoke_test  # tiny random model, ~1 min for the whole file

PROMPTS = ["the quick brown fox", "the earth is round"]


def tiny_gemma3(seed=0, outliers=True, dtype="float32"):
    """Small random text-only Gemma3 with a few outlier channels in the pre-attention / pre-FFW norms."""
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
        dtype=dtype,
    )
    model = Gemma3CausalLM(backbone=backbone, preprocessor=preprocessor)
    if outliers:
        rng = np.random.default_rng(seed)
        for block in backbone.transformer_layers:
            for norm in (block.pre_attention_norm, block.pre_ffw_norm):
                g = np.asarray(norm.scale).astype(np.float32)
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


def test_save_load_keeps_smooth_quant_params(tmp_path):
    """The SmoothQuant options are stored with the model like the other options (so it's visible that it was applied);
    loading only prepares the quantized layers (no smoothing or calibration) and gives the same outputs."""
    model, preprocessor = tiny_gemma3()
    config = StaticQuantConfig(const_scale=True, const_weight=True, smooth_quant=True, smooth_quant_alpha=0.4)
    qmodel = quantize_model(model, config, generate_calibration)
    inputs = preprocessor.generate_preprocess(PROMPTS)
    inputs = {k: inputs[k] for k in ("token_ids", "padding_mask")}
    expected = np.asarray(qmodel.backbone(inputs))
    # the mock tokenizer has no assets to save, so save the quantized backbone (it carries the quant config)
    qmodel.backbone.save_to_preset(str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())
    saved_quant = saved["config"]["_quant_config"]["config"]
    assert saved_quant["smooth_quant"] is True and saved_quant["smooth_quant_alpha"] == 0.4
    loaded = Backbone.from_preset(str(tmp_path))
    np.testing.assert_allclose(np.asarray(loaded(inputs)), expected, rtol=1e-5, atol=1e-5)


def _norm_scales(model):
    return [
        (np.asarray(b.pre_attention_norm.scale).copy(), np.asarray(b.pre_ffw_norm.scale).copy())
        for b in model.backbone.transformer_layers
    ]


def test_composable_config_smooths_only_selected_blocks():
    """A composable config can enable SmoothQuant for part of the model; the other blocks are quantized unsmoothed."""
    model, preprocessor = tiny_gemma3()
    ref = logits(model, preprocessor)
    before = _norm_scales(model)
    config = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8") + StaticQuantConfig(
        weight_dtype="int8", activation_dtype="int8", smooth_quant=True, white_list=[r"decoder_block_0/"]
    )
    qmodel = quantize_model(model, config, generate_calibration)
    after = _norm_scales(qmodel)
    assert not np.allclose(after[0][0], before[0][0]) and not np.allclose(after[0][1], before[0][1])  # block 0
    np.testing.assert_array_equal(after[1][0], before[1][0])  # block 1 untouched
    np.testing.assert_array_equal(after[1][1], before[1][1])
    assert rel_err(logits(qmodel, preprocessor), ref) < 0.1


def test_group_with_conflicting_settings_is_skipped():
    """Projections sharing an input must agree: smoothing only the query (not key/value) skips the q/k/v group."""
    model, _ = tiny_gemma3()
    before = _norm_scales(model)
    config = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8") + StaticQuantConfig(
        weight_dtype="int8", activation_dtype="int8", smooth_quant=True, white_list=[r"decoder_block_0/attention/query"]
    )
    quantize_model(model, config, generate_calibration)
    np.testing.assert_array_equal(_norm_scales(model)[0][0], before[0][0])


def test_per_layer_settings_resolution():
    model, _ = tiny_gemma3()
    q_and_k = {"query", "key"}

    def settings(layer):  # q/k with alpha 0.4, v not quantized, gate/up/down/o without SmoothQuant
        if layer.name == "value":
            return None
        return (True, 0.4) if layer.name in q_and_k else (False, 0.5)

    groups = sq.find_smooth_groups(model, settings)
    assert sorted(g.name for g in groups) == ["decoder_block_0/qkv", "decoder_block_1/qkv"]
    assert all(g.alpha == 0.4 for g in groups)


def test_bfloat16_fold_is_exact():
    """In bfloat16, a folded norm gain (1 + scale') close to 0 is coarse; the consumers are scaled by the factors the
    stored gain implements, so gain x weight stays equal per input channel (up to the bf16 rounding of the weights)."""
    model, preprocessor = tiny_gemma3(dtype="bfloat16")
    block = model.backbone.transformer_layers[0]
    gain0 = 1.0 + np.asarray(block.pre_attention_norm.scale).astype(np.float32)
    q0 = np.asarray(block.attention.query_dense.kernel).astype(np.float32)  # (heads, in, head_dim)
    x = preprocessor.generate_preprocess(PROMPTS)
    sq.apply_smooth_quant(model, lambda m: m(x))
    gain1 = 1.0 + np.asarray(block.pre_attention_norm.scale).astype(np.float32)
    q1 = np.asarray(block.attention.query_dense.kernel).astype(np.float32)
    assert (gain0 / gain1).max() > 10  # strong smoothing on the outlier channels, where bf16 rounding matters
    np.testing.assert_allclose(gain1[None, :, None] * q1, gain0[None, :, None] * q0, rtol=2e-2, atol=1e-6)


def test_observer_keeps_input_dtype():
    """The observer must not autocast: the observed projection gets its bf16 input unchanged during calibration."""
    x = jnp.ones((2, 3, 4), dtype="bfloat16")
    for kwargs in ({}, {"dtype": "float32"}):
        obs = sq.ChannelAbsMaxObserver((4,), **kwargs)
        assert obs(x).dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(obs.channel_abs_max), np.ones(4))


def test_config_expand():
    """StaticQuantConfig keeps an explicit signature, so BaseConfig.expand works (with and without SmoothQuant)."""
    assert len(StaticQuantConfig().expand()) == 1
    configs = StaticQuantConfig(smooth_quant=True, smooth_quant_alpha=[0.3, 0.5]).expand()
    assert sorted(c.smooth_quant_alpha for c in configs) == [0.3, 0.5]


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

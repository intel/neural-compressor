"""The quantized KV cache of QStaticCachedGemma3Attention must not change results.

QStaticCachedGemma3Attention.call stores the KV cache quantized and quantizes only the new
token each step, instead of keeping a bf16 cache and re-quantizing all of it on every step.
With a static per-tensor scale quantization is elementwise, so the two must agree exactly:
same cache contents, same logits, same generated tokens.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import keras
import numpy as np
import pytest
from jax import numpy as jnp
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM

from neural_compressor.jax import StaticQuantConfig, quantize_model
from neural_compressor.jax.quantization.layers_static import QStaticCachedGemma3Attention

BATCH, PROMPT, MAX_LEN = 3, 6, 14


def _quantized_lm(activation_dtype):
    keras.utils.set_random_seed(0)
    backbone = Gemma3Backbone(
        vocabulary_size=64,
        image_size=16,
        num_layers=2,
        num_query_heads=4,
        num_key_value_heads=2,
        hidden_dim=32,
        intermediate_dim=64,
        head_dim=8,
        # Short window so decode crosses it and the sliding-window mask path is exercised.
        use_sliding_window_attention=True,
        sliding_window_size=4,
        vision_encoder=None,
        dtype="bfloat16",
    )
    lm = Gemma3CausalLM(preprocessor=None, backbone=backbone)
    rng = np.random.default_rng(0)
    calib = {
        "token_ids": rng.integers(1, 64, size=(BATCH, MAX_LEN)).astype("int32"),
        "padding_mask": np.ones((BATCH, MAX_LEN), dtype=bool),
    }
    config = StaticQuantConfig(
        weight_dtype=activation_dtype, activation_dtype=activation_dtype, const_scale=True
    )
    return quantize_model(lm, config, lambda m: m(calib), inplace=True)


def _inputs():
    rng = np.random.default_rng(1)
    token_ids = np.zeros((BATCH, MAX_LEN), dtype="int32")
    token_ids[:, :PROMPT] = rng.integers(1, 64, size=(BATCH, PROMPT))
    padding_mask = np.zeros((BATCH, MAX_LEN), dtype=bool)
    padding_mask[:, :PROMPT] = True
    padding_mask[0, PROMPT - 2 :] = False  # uneven prompt lengths
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def _prefill_then_decode(lm, inputs, steps=4):
    """Prefill, then a few single-token decode steps. Returns (logits per call, final cache)."""
    token_ids = inputs["token_ids"]
    shape = [BATCH, MAX_LEN, lm.backbone.num_key_value_heads, lm.backbone.head_dim]
    cache = tuple(
        (jnp.zeros(shape, lm.compute_dtype), jnp.zeros(shape, lm.compute_dtype))
        for _ in range(lm.backbone.num_layers)
    )
    logits, _, cache = lm.call_with_cache(
        token_ids, cache, 0, padding_mask=inputs["padding_mask"]
    )
    all_logits = [np.asarray(logits.astype("float32"))]
    for i in range(PROMPT, PROMPT + steps):
        logits, _, cache = lm.call_with_cache(token_ids[:, i : i + 1], cache, i)
        all_logits.append(np.asarray(logits.astype("float32")))
    return all_logits, cache


@pytest.mark.parametrize("activation_dtype", ["fp8_e4m3", "int8"])
def test_quantized_kv_cache_matches_requantized_bf16_cache(activation_dtype, monkeypatch):
    lm = _quantized_lm(activation_dtype)
    inputs = _inputs()
    attn = lm.backbone.transformer_layers[0].attention
    assert isinstance(attn, QStaticCachedGemma3Attention)

    new_logits, new_cache = _prefill_then_decode(lm, inputs)
    new_tokens = np.asarray(lm.generate_step(inputs)["token_ids"])

    # The cache really is carried quantized, not in the compute dtype.
    assert new_cache[0][0].dtype == attn.k_qdq.storage_dtype
    assert new_cache[0][1].dtype == attn.v_qdq.storage_dtype

    # Reference: the original bf16-cache call, which re-quantizes the whole cache every step.
    monkeypatch.setattr(QStaticCachedGemma3Attention, "call", CachedGemma3Attention.call)
    ref_logits, ref_cache = _prefill_then_decode(lm, inputs)
    ref_tokens = np.asarray(lm.generate_step(inputs)["token_ids"])
    assert ref_cache[0][0].dtype == lm.compute_dtype

    for new, ref in zip(new_logits, ref_logits):
        np.testing.assert_array_equal(new, ref)
    np.testing.assert_array_equal(new_tokens, ref_tokens)
    layers = lm.backbone.transformer_layers
    for layer, (nk, nv), (rk, rv) in zip(layers, new_cache, ref_cache):
        a = layer.attention
        np.testing.assert_array_equal(
            np.asarray(a.k_qdq.dequantize_from_storage(nk).astype("float32")),
            np.asarray(a.k_qdq(rk).astype("float32")),
        )
        np.testing.assert_array_equal(
            np.asarray(a.v_qdq.dequantize_from_storage(nv).astype("float32")),
            np.asarray(a.v_qdq(rv).astype("float32")),
        )


@pytest.mark.parametrize("activation_dtype", ["fp8_e4m3", "int8"])
def test_quantized_kv_cache_jitted_generate(activation_dtype, monkeypatch):
    # The compiled generate loop carries the quantized cache through a while loop and
    # updates it under `cache_update_mask`; the eager helpers above cover neither.
    lm = _quantized_lm(activation_dtype)
    inputs = _inputs()

    lm.make_generate_function()
    new_tokens = np.asarray(lm.generate_function(inputs)["token_ids"])

    monkeypatch.setattr(QStaticCachedGemma3Attention, "call", CachedGemma3Attention.call)
    lm.make_generate_function()
    ref_tokens = np.asarray(lm.generate_function(inputs)["token_ids"])

    np.testing.assert_array_equal(new_tokens, ref_tokens)


def test_quantized_cache_update_saturates():
    # Decode can produce values beyond the calibrated range. inc.quantize clamps them to
    # the fp8 range; a quantize fused into the value projection must do the same rather
    # than overflow to NaN (e4m3fn has no inf), which would poison the cache.
    lm = _quantized_lm("fp8_e4m3")
    attn = lm.backbone.transformer_layers[0].attention
    x = np.random.default_rng(2).normal(size=(BATCH, 1, 32)) * 50
    x = jnp.asarray(x, dtype=lm.compute_dtype)

    stored = jax.jit(lambda x: attn.v_qdq.quantize_to_storage(attn.value_dense(x)))(x)
    values = np.asarray(attn.v_qdq.dequantize_from_storage(stored).astype("float32"))

    assert not np.isnan(values).any()
    limit = np.abs(values).max()
    assert np.sum(np.abs(values) == limit) > 1  # out-of-range values saturated to the limit

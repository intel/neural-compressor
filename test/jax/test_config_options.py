import keras
import pytest
from jax import numpy as jnp

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model

# Mark all tests in this file as smoke tests
pytestmark = pytest.mark.smoke_test


def test_weight_quantization_granularity():
    inputs = keras.Input(shape=(8,))
    x = keras.layers.Dense(16, use_bias=False, name="pc_1")(inputs)
    x = keras.layers.Dense(8, use_bias=False, name="pt_1")(x)
    x = keras.layers.Dense(12, use_bias=False, name="pc_2")(x)
    outputs = keras.layers.Dense(6, use_bias=False, name="pt_2")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="test_model")

    # Build once so layer paths are populated for regex filtering.
    _ = model(jnp.ones((1, 8), dtype=jnp.float32))

    per_channel_cfg = DynamicQuantConfig(
        weight_dtype="int8",
        activation_dtype="int8",
        weight_scale_granularity="per_channel",
        white_list=["pc_.*"],
    )
    per_tensor_cfg = DynamicQuantConfig(
        weight_dtype="int8",
        activation_dtype="int8",
        weight_scale_granularity="per_tensor",
        white_list=["pt_.*"],
    )

    q_model = quantize_model(model, per_channel_cfg + per_tensor_cfg)

    def _layer_by_name(model, name):
        for layer in model._flatten_layers():
            if layer.name == name:
                return layer
        raise AssertionError(f"Layer {name!r} not found")

    for layer_name in ["pc_1", "pc_2"]:
        assert (
            _layer_by_name(q_model, layer_name).w_scale.value.size > 1
        ), f"Expected per-channel scale for {layer_name!r} to have multiple values"

    for layer_name in ["pt_1", "pt_2"]:
        assert (
            _layer_by_name(q_model, layer_name).w_scale.value.size == 1
        ), f"Expected per-tensor scale for {layer_name!r} to have exactly one value"


def _build_mha_model():
    inputs = keras.Input(shape=(4, 8))
    outputs = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, name="mha")(inputs, inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mha_model")
    # Build once so layer paths are populated and weights are initialized.
    _ = model(jnp.ones((1, 4, 8), dtype=jnp.float32))
    return model


@pytest.mark.parametrize("dynamic", [False, True], ids=["dynamic=False", "dynamic=True"])
@pytest.mark.parametrize("dot_product_attention_enable", [False, True], ids=["dpa_enable=False", "dpa_enable=True"])
def test_dot_product_attention_enable(monkeypatch, dynamic, dot_product_attention_enable):
    """Fused dot_product_attention path is taken only when the option is enabled.

    A MultiHeadAttention layer routes through ``ops.dot_product_attention`` when
    ``dot_product_attention_enable`` is True and falls back to the einsum path
    otherwise. This is verified by spying on ``keras.ops.dot_product_attention``.
    """
    model = _build_mha_model()
    sample = jnp.ones((1, 4, 8), dtype=jnp.float32)

    calls = {"count": 0}
    real_dpa = keras.ops.dot_product_attention

    def _spy(*args, **kwargs):
        calls["count"] += 1
        return real_dpa(*args, **kwargs)

    monkeypatch.setattr(keras.ops, "dot_product_attention", _spy)

    common = dict(
        weight_dtype="int8",
        activation_dtype="int8",
        dot_product_attention_enable=dot_product_attention_enable,
    )
    if dynamic:
        q_model = quantize_model(model, DynamicQuantConfig(**common))
    else:
        q_model = quantize_model(model, StaticQuantConfig(**common), lambda m: m(sample))

    _ = q_model(sample)

    if dot_product_attention_enable:
        assert calls["count"] > 0, "Expected the fused dot_product_attention path to be used"
    else:
        assert calls["count"] == 0, "Expected the einsum fallback path (no dot_product_attention)"

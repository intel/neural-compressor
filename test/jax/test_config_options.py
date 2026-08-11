import keras
import pytest
from jax import numpy as jnp

from neural_compressor.jax import DynamicQuantConfig, quantize_model

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

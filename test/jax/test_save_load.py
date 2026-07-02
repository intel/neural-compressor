#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for JAX quantization save/load functionality.

This test suite validates that quantized models can be properly saved and loaded,
preserving both the quantization configuration and the quantized behavior.
"""

import os
import tempfile

import keras
import pytest
from jax import numpy as jnp

from neural_compressor.common.base_config import ComposableConfig
from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model

# Mark all tests in this file as smoke tests
pytestmark = pytest.mark.smoke_test


@pytest.mark.parametrize(
    "weight_dtype, activation_dtype, const_scale, const_weight",
    [("fp8_e4m3", "fp8_e4m3", False, False), ("fp8_e5m2", "fp8_e5m2", False, True), ("int8", "int8", False, True)],
)
class TestDynamicQuantSaveLoad:
    """Test save/load for DynamicQuantConfig."""

    def test_dynamic_quant_save_load(self, weight_dtype, activation_dtype, const_scale, const_weight, model, test_data):
        """Test save/load with DynamicQuantConfig."""
        # Quantize model
        config = DynamicQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
        )
        q_model = quantize_model(model, config)

        # Get output before save
        output_before = q_model.predict(test_data)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        # Get output after load
        output_after = loaded_model.predict(test_data)

        # Verify outputs match
        assert jnp.allclose(
            output_before, output_after, rtol=1e-5
        ), f"Output mismatch: before={output_before}, after={output_after}"

        # Verify quantization config is preserved
        assert hasattr(loaded_model, "_quant_config"), "Loaded model missing _quant_config"
        for attribute in ["name"] + config.__class__.params_list:
            expected_value = getattr(config, attribute)
            actual_value = getattr(loaded_model._quant_config, attribute)
            assert (
                actual_value == expected_value
            ), f"{attribute} mismatch: expected={expected_value}, got={actual_value}"


@pytest.mark.parametrize(
    "weight_dtype, activation_dtype, const_scale, const_weight",
    [
        ("fp8_e4m3", "fp8_e4m3", False, False),
        ("fp8_e4m3", "fp8_e4m3", True, False),
        ("fp8_e5m2", "fp8_e5m2", True, True),
        ("int8", "int8", True, True),
    ],
)
class TestStaticQuantSaveLoad:
    """Test save/load for StaticQuantConfig."""

    def test_static_quant_save_load(
        self, weight_dtype, activation_dtype, const_scale, const_weight, model, calibration_data, test_data
    ):
        """Test save/load with StaticQuantConfig."""
        # Quantize model
        config = StaticQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
        )

        def calib_fn(m):
            return m(calibration_data)

        q_model = quantize_model(model, config, calib_function=calib_fn)

        # Get output before save
        output_before = q_model.predict(test_data)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        # Get output after load
        output_after = loaded_model.predict(test_data)

        # Verify outputs match
        assert jnp.allclose(
            output_before, output_after, rtol=1e-5
        ), f"Output mismatch: before={output_before}, after={output_after}"

        # Verify quantization config is preserved
        assert hasattr(loaded_model, "_quant_config"), "Loaded model missing _quant_config"
        for attribute in ["name"] + config.__class__.params_list:
            expected_value = getattr(config, attribute)
            actual_value = getattr(loaded_model._quant_config, attribute)
            assert (
                actual_value == expected_value
            ), f"{attribute} mismatch: expected={expected_value}, got={actual_value}"


class TestMultipleRoundTrips:
    """Test multiple save/load round trips."""

    def test_dynamic_quant_multiple_saves(self, model, test_data):
        """Test that model can be saved and loaded multiple times."""
        # Quantize model
        config = DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3")
        q_model = quantize_model(model, config)

        # Get initial output
        output_initial = q_model.predict(test_data)

        # Save and load multiple times
        current_model = q_model
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, f"model_{i}.keras")
                keras.saving.save_model(current_model, save_path)
                current_model = keras.saving.load_model(save_path)

            # Verify output matches initial
            output_current = current_model.predict(test_data)
            assert jnp.allclose(output_initial, output_current, rtol=1e-5), f"Output mismatch after round trip {i+1}"

            # Verify config is preserved
            assert current_model._quant_config.name == config.name, f"Config name changed after round trip {i+1}"
            assert (
                current_model._quant_config.weight_dtype == config.weight_dtype
            ), f"weight_dtype changed after round trip {i+1}"
            assert (
                current_model._quant_config.activation_dtype == config.activation_dtype
            ), f"activation_dtype changed after round trip {i+1}"
            assert (
                current_model._quant_config.const_scale == config.const_scale
            ), f"const_scale changed after round trip {i+1}"
            assert (
                current_model._quant_config.const_weight == config.const_weight
            ), f"const_weight changed after round trip {i+1}"

    def test_static_quant_multiple_saves(self, model, calibration_data, test_data):
        """Test that static quantized model can be saved and loaded multiple times."""
        # Quantize model
        config = StaticQuantConfig(weight_dtype="fp8_e5m2", activation_dtype="fp8_e5m2")

        def calib_fn(m):
            return m(calibration_data)

        q_model = quantize_model(model, config, calib_function=calib_fn)

        # Get initial output
        output_initial = q_model.predict(test_data)

        # Save and load multiple times
        current_model = q_model
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, f"model_{i}.keras")
                keras.saving.save_model(current_model, save_path)
                current_model = keras.saving.load_model(save_path)

            # Verify output matches initial
            output_current = current_model.predict(test_data)
            assert jnp.allclose(output_initial, output_current, rtol=1e-5), f"Output mismatch after round trip {i+1}"

            # Verify config is preserved
            assert current_model._quant_config.name == config.name, f"Config name changed after round trip {i+1}"
            assert (
                current_model._quant_config.weight_dtype == config.weight_dtype
            ), f"weight_dtype changed after round trip {i+1}"
            assert (
                current_model._quant_config.activation_dtype == config.activation_dtype
            ), f"activation_dtype changed after round trip {i+1}"
            assert (
                current_model._quant_config.const_scale == config.const_scale
            ), f"const_scale changed after round trip {i+1}"
            assert (
                current_model._quant_config.const_weight == config.const_weight
            ), f"const_weight changed after round trip {i+1}"


def _find_layer(model, name):
    """Return the (possibly quantized) sub-layer with the given name."""
    for layer in model._flatten_layers(recursive=True):
        if layer.name == name:
            return layer
    raise AssertionError(f"Layer {name!r} not found")


def _quantize_composable(model, calibration_data):
    """Quantize the shared model with a static + dynamic ComposableConfig."""
    # Static quant on 'first', dynamic quant on 'second'.
    config = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", include=["first"]) + DynamicQuantConfig(
        weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", include=["second"]
    )

    def calib_fn(m):
        return m(calibration_data)

    return quantize_model(model, config, calib_function=calib_fn)


def _quantize_three_way_composable(model, calibration_data):
    """Quantize with a ``static + dynamic + static`` config with overlapping rules.

    Per-layer resolution follows "last matching sub-config wins":
    ``first`` -> static, ``second`` -> dynamic, ``third`` -> static.
    """
    config = (
        StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", include=["first", "second"])
        + DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", include=["second", "third"])
        + StaticQuantConfig(
            weight_dtype="int8", activation_dtype="int8", include=["Dense"], exclude=["first", "second"]
        )
    )

    def calib_fn(m):
        return m(calibration_data)

    return quantize_model(model, config, calib_function=calib_fn)


class TestComposableConfigSaveLoad:
    """Test save/load for a ComposableConfig (static + dynamic sub-configs)."""

    def test_composable_quant_save_load_preserves_output(self, model, calibration_data, test_data):
        q_model = _quantize_composable(model, calibration_data)
        output_before = q_model.predict(test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        output_after = loaded_model.predict(test_data)
        assert jnp.allclose(
            output_before, output_after, rtol=1e-5
        ), f"Output mismatch: before={output_before}, after={output_after}"

    def test_composable_quant_config_is_preserved(self, model, calibration_data):
        q_model = _quantize_composable(model, calibration_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        assert hasattr(loaded_model, "_quant_config"), "Loaded model missing _quant_config"
        loaded_config = loaded_model._quant_config
        assert isinstance(loaded_config, ComposableConfig), "Loaded config must remain a ComposableConfig"
        assert len(loaded_config.config_list) == 2, "Both sub-configs must survive save/load"

        static_cfg, dynamic_cfg = loaded_config.config_list
        assert static_cfg.name == "static_quant", "First sub-config must be the static one"
        assert static_cfg.weight_dtype == "int8", "Static sub-config weight_dtype must be preserved"
        assert static_cfg.include == ["first"], "Static sub-config include filter must be preserved"
        assert dynamic_cfg.name == "dynamic_quant", "Second sub-config must be the dynamic one"
        assert dynamic_cfg.weight_dtype == "fp8_e4m3", "Dynamic sub-config weight_dtype must be preserved"
        assert dynamic_cfg.include == ["second"], "Dynamic sub-config include filter must be preserved"

    def test_composable_layer_classes_after_load(self, model, calibration_data):
        # Regression for the per-layer deserialization fallback: each layer must
        # be restored with the correct quantized class.
        q_model = _quantize_composable(model, calibration_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        assert (
            type(_find_layer(loaded_model, "first")).__name__ == "QStaticDense"
        ), "'first' must reload as a static layer"
        assert (
            type(_find_layer(loaded_model, "second")).__name__ == "QDynamicDense"
        ), "'second' must reload as a dynamic layer"


class TestComplexComposableConfigSaveLoad:
    """Save/load for a three-way ``static + dynamic + static`` ComposableConfig
    with overlapping include/exclude rules."""

    def test_output_is_preserved(self, model, calibration_data, test_data):
        q_model = _quantize_three_way_composable(model, calibration_data)
        output_before = q_model.predict(test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        output_after = loaded_model.predict(test_data)
        assert jnp.allclose(
            output_before, output_after, rtol=1e-5
        ), f"Output mismatch: before={output_before}, after={output_after}"

    def test_config_is_preserved(self, model, calibration_data):
        q_model = _quantize_three_way_composable(model, calibration_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        assert hasattr(loaded_model, "_quant_config"), "Loaded model missing _quant_config"
        loaded_config = loaded_model._quant_config
        assert isinstance(loaded_config, ComposableConfig), "Loaded config must remain a ComposableConfig"
        assert len(loaded_config.config_list) == 3, "All three sub-configs must survive save/load"

        static_a, dynamic, static_b = loaded_config.config_list
        assert [c.name for c in loaded_config.config_list] == [
            "static_quant",
            "dynamic_quant",
            "static_quant",
        ], "Sub-config order (static, dynamic, static) must be preserved"
        assert static_a.include == ["first", "second"], "First static sub-config include must be preserved"
        assert dynamic.include == ["second", "third"], "Dynamic sub-config include must be preserved"
        assert static_b.include == ["Dense"], "Second static sub-config include must be preserved"
        assert static_b.exclude == ["first", "second"], "Second static sub-config exclude must be preserved"

    def test_layer_classes_after_load(self, model, calibration_data):
        q_model = _quantize_three_way_composable(model, calibration_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.keras")
            keras.saving.save_model(q_model, save_path)
            loaded_model = keras.saving.load_model(save_path)

        assert (
            type(_find_layer(loaded_model, "first")).__name__ == "QStaticDense"
        ), "'first' must reload as static (last-match-wins consistent with quantization)"
        assert (
            type(_find_layer(loaded_model, "second")).__name__ == "QDynamicDense"
        ), "'second' must reload as dynamic (last-match-wins consistent with quantization)"
        assert (
            type(_find_layer(loaded_model, "third")).__name__ == "QStaticDense"
        ), "'third' must reload as static (last-match-wins consistent with quantization)"

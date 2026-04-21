#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for JAX quantization save/load functionality.

This test suite validates that quantized models can be properly saved and loaded,
preserving both the quantization configuration and the quantized behavior.
"""

import os
import tempfile

import pytest

os.environ["KERAS_BACKEND"] = "jax"

import keras
from jax import numpy as jnp

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model


@keras.saving.register_keras_serializable()
class SimpleModel(keras.Model):
    """A simple model for testing quantization save/load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(
            4, activation="relu", use_bias=True, kernel_initializer=keras.initializers.random_normal(seed=2000)
        )
        self.dense2 = keras.layers.Dense(
            2, activation="linear", use_bias=True, kernel_initializer=keras.initializers.random_normal(seed=2000)
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def create_simple_model():
    """Create a simple model for testing."""
    model = SimpleModel()
    # Initialize model with dummy data
    _ = model(jnp.ones((1, 8)))
    return model


def create_calibration_data():
    """Create calibration data for static quantization."""
    return jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )


def create_test_data():
    """Create test data for inference."""
    return jnp.array(
        [
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            [7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5],
        ]
    )


@pytest.mark.parametrize(
    "weight_dtype, activation_dtype, const_scale, const_weight",
    [("fp8_e4m3", "fp8_e4m3", False, False), ("fp8_e5m2", "fp8_e5m2", False, True), ("int8", "int8", False, True)],
)
class TestDynamicQuantSaveLoad:
    """Test save/load for DynamicQuantConfig."""

    def test_dynamic_quant_save_load(self, weight_dtype, activation_dtype, const_scale, const_weight):
        """Test save/load with DynamicQuantConfig."""
        model = create_simple_model()
        test_data = create_test_data()

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

    def test_static_quant_save_load(self, weight_dtype, activation_dtype, const_scale, const_weight):
        """Test save/load with StaticQuantConfig."""
        model = create_simple_model()
        calib_data = create_calibration_data()
        test_data = create_test_data()

        # Quantize model
        config = StaticQuantConfig(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            const_scale=const_scale,
            const_weight=const_weight,
        )

        def calib_fn(m):
            return m(calib_data)

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

    def test_dynamic_quant_multiple_saves(self):
        """Test that model can be saved and loaded multiple times."""
        model = create_simple_model()
        test_data = create_test_data()

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
            assert current_model._quant_config.name == config.name
            assert current_model._quant_config.weight_dtype == config.weight_dtype
            assert current_model._quant_config.activation_dtype == config.activation_dtype
            assert current_model._quant_config.const_scale == config.const_scale
            assert current_model._quant_config.const_weight == config.const_weight

    def test_static_quant_multiple_saves(self):
        """Test that static quantized model can be saved and loaded multiple times."""
        model = create_simple_model()
        calib_data = create_calibration_data()
        test_data = create_test_data()

        # Quantize model
        config = StaticQuantConfig(weight_dtype="fp8_e5m2", activation_dtype="fp8_e5m2")

        def calib_fn(m):
            return m(calib_data)

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
            assert current_model._quant_config.name == config.name
            assert current_model._quant_config.weight_dtype == config.weight_dtype
            assert current_model._quant_config.activation_dtype == config.activation_dtype
            assert current_model._quant_config.const_scale == config.const_scale
            assert current_model._quant_config.const_weight == config.const_weight

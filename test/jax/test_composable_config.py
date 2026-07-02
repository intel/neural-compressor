#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ComposableConfig quantization.

These tests validate that composing a ``StaticQuantConfig`` with a
``DynamicQuantConfig`` (each targeting a different subset of layers) produces a
model in which every layer is quantized with the expected scheme and remains
numerically well-behaved.

The shared ``model`` / ``calibration_data`` / ``test_data`` fixtures (see
conftest.py) supply a model with named Dense layers ``first``/``second``/``third``.
"""

from jax import numpy as jnp

from neural_compressor.common.base_config import ComposableConfig
from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.quantization.quantize import _build_configs_mapping_composable


def _calib_fn(calibration_data):
    """Build a calibration function that feeds ``calibration_data`` to a model."""

    def _fn(model):
        _ = model(calibration_data)

    return _fn


def _read_scale(layer):
    """Return an activation scale value regardless of const-scale mode."""
    scale = layer.a_scale
    return scale.value if hasattr(scale, "value") else scale


def _layer_by_name(model, name):
    for layer in model._flatten_layers(recursive=True):
        if layer.name == name:
            return layer
    raise AssertionError(f"Layer {name!r} not found")


class TestComposition:
    def test_static_plus_dynamic_creates_composable(self):
        static = StaticQuantConfig(include=["first"])
        dynamic = DynamicQuantConfig(include=["second"])
        composed = static + dynamic
        assert isinstance(
            composed, ComposableConfig
        ), "Adding two different config types must produce a ComposableConfig"
        assert len(composed.config_list) == 2, "Composable must hold both sub-configs"
        assert composed.config_list[0] is static, "First sub-config must be the static one (left operand)"
        assert composed.config_list[1] is dynamic, "Second sub-config must be the dynamic one (right operand)"

    def test_build_configs_mapping_assigns_expected_schemes(self, model):
        composed = StaticQuantConfig(include=["first"]) + DynamicQuantConfig(include=["second"])
        mapping = _build_configs_mapping_composable(model, composed)

        static_ids = [op for (op, _), cfg in mapping.items() if cfg.name == "static_quant"]
        dynamic_ids = [op for (op, _), cfg in mapping.items() if cfg.name == "dynamic_quant"]
        assert any("first" in op for op in static_ids), "'first' must be assigned the static config"
        assert any("second" in op for op in dynamic_ids), "'second' must be assigned the dynamic config"
        assert all("third" not in op for (op, _) in mapping), "'third' is unmatched and must be absent from the mapping"

    def test_build_configs_mapping_last_config_wins_on_conflict(self, model):
        # Both sub-configs target the same layer; the later one must win.
        composed = StaticQuantConfig(include=["second"]) + DynamicQuantConfig(include=["second"])
        mapping = _build_configs_mapping_composable(model, composed)
        second_cfgs = [cfg for (op, _), cfg in mapping.items() if "second" in op]
        assert len(second_cfgs) == 1, "'second' must have exactly one resolved config"
        assert second_cfgs[0].name == "dynamic_quant", "On conflict the later (dynamic) config must win"


class TestComposableQuantization:
    def test_layers_use_expected_quant_classes(self, model, calibration_data):
        composed = StaticQuantConfig(include=["first"]) + DynamicQuantConfig(include=["second"])
        q_model = quantize_model(model, composed, calib_function=_calib_fn(calibration_data), inplace=True)

        assert type(_layer_by_name(q_model, "first")).__name__ == "QStaticDense", "'first' must be statically quantized"
        assert (
            type(_layer_by_name(q_model, "second")).__name__ == "QDynamicDense"
        ), "'second' must be dynamically quantized"
        # 'third' was not targeted by either config and must remain untouched.
        assert (
            type(_layer_by_name(q_model, "third")).__name__ == "Dense"
        ), "'third' was untargeted and must stay an unquantized Dense"

    def test_output_is_finite(self, model, calibration_data, test_data):
        composed = StaticQuantConfig(include=["first"]) + DynamicQuantConfig(include=["second"])
        q_model = quantize_model(model, composed, calib_function=_calib_fn(calibration_data), inplace=True)
        output = q_model(test_data)
        assert output.shape == (test_data.shape[0], 2), "Quantized model must preserve the output shape"
        assert bool(jnp.all(jnp.isfinite(output))), "Quantized model output must be finite (no NaN/inf)"

    def test_static_algorithm_runs_before_dynamic(self, model, calibration_data):
        """Static calibration must run on the original (un-quantized) model.

        A static layer placed *after* a dynamically-quantized layer must produce
        the same calibrated activation scale as when only static quantization is
        applied. If the dynamic algorithm ran first, it would perturb the static
        layer's input activations and change its scale.
        """
        calib = _calib_fn(calibration_data)

        # Sum order is dynamic-then-static on purpose: static must still run
        # first regardless of composition order. Both runs start from the same
        # FP32 model (inplace=False clones it) so their weights are identical.
        composed = DynamicQuantConfig(include=["first"]) + StaticQuantConfig(include=["second"])
        q_composed = quantize_model(model, composed, calib_function=calib, inplace=False)

        static_only = StaticQuantConfig(include=["second"])
        q_static = quantize_model(model, static_only, calib_function=calib, inplace=False)

        composed_scale = _read_scale(_layer_by_name(q_composed, "second"))
        static_scale = _read_scale(_layer_by_name(q_static, "second"))
        assert jnp.allclose(
            jnp.array(composed_scale), jnp.array(static_scale), rtol=1e-5
        ), f"Activation scale differs: composed={composed_scale}, static_only={static_scale}"

    def test_per_layer_dtypes_are_honored(self, model, calibration_data, test_data):
        composed = StaticQuantConfig(
            weight_dtype="int8", activation_dtype="int8", include=["first"]
        ) + DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", include=["second"])
        q_model = quantize_model(model, composed, calib_function=_calib_fn(calibration_data), inplace=True)

        assert (
            type(_layer_by_name(q_model, "first")).__name__ == "QStaticDense"
        ), "'first' must use the static int8 scheme"
        assert (
            type(_layer_by_name(q_model, "second")).__name__ == "QDynamicDense"
        ), "'second' must use the dynamic fp8 scheme"
        output = q_model(test_data)
        assert bool(jnp.all(jnp.isfinite(output))), "Mixed-dtype quantized model output must be finite"


def _three_way_config():
    """Compose ``static + dynamic + static`` with deliberately overlapping rules.

    Per-layer resolution follows the "last matching sub-config wins" rule of
    ``_build_configs_mapping_composable``:

    * ``first``  -> only ``static_a`` matches                -> static
    * ``second`` -> ``static_a`` and ``dynamic`` match       -> dynamic (later)
    * ``third``  -> ``dynamic`` and ``static_b`` match       -> static  (later)
    """
    static_a = StaticQuantConfig(weight_dtype="int8", activation_dtype="int8", include=["first", "second"])
    dynamic = DynamicQuantConfig(weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", include=["second", "third"])
    # include everything but exclude the layers claimed earlier -> resolves to 'third'.
    static_b = StaticQuantConfig(
        weight_dtype="int8", activation_dtype="int8", include=["Dense"], exclude=["first", "second"]
    )
    return static_a + dynamic + static_b


class TestThreeWayComposition:
    """``static + dynamic + static`` with overlapping include/exclude rules."""

    def test_composition_preserves_three_configs_in_order(self):
        composed = _three_way_config()
        assert isinstance(composed, ComposableConfig), "Chaining three configs must yield a ComposableConfig"
        assert len(composed.config_list) == 3, "All three sub-configs must be retained"
        assert [cfg.name for cfg in composed.config_list] == [
            "static_quant",
            "dynamic_quant",
            "static_quant",
        ], "Sub-config order must match the composition order (static, dynamic, static)"

    def test_overlapping_rules_resolve_last_match_wins(self, model):
        mapping = _build_configs_mapping_composable(model, _three_way_config())
        resolved = {op.split("/")[-1]: cfg.name for (op, _), cfg in mapping.items()}
        assert resolved == {
            "first": "static_quant",
            "second": "dynamic_quant",
            "third": "static_quant",
        }, "Overlapping rules must resolve last-match-wins: first->static, second->dynamic, third->static"

    def test_quantized_layer_classes_and_output(self, model, calibration_data, test_data):
        q_model = quantize_model(model, _three_way_config(), calib_function=_calib_fn(calibration_data), inplace=True)

        assert type(_layer_by_name(q_model, "first")).__name__ == "QStaticDense", "'first' resolves to static"
        assert (
            type(_layer_by_name(q_model, "second")).__name__ == "QDynamicDense"
        ), "'second' resolves to dynamic (last match)"
        assert (
            type(_layer_by_name(q_model, "third")).__name__ == "QStaticDense"
        ), "'third' resolves to static (last match)"

        output = q_model(test_data)
        assert output.shape == (test_data.shape[0], 2), "Three-way quantized model must preserve output shape"
        assert bool(jnp.all(jnp.isfinite(output))), "Three-way quantized model output must be finite"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for quantization config white_list/exclude_list filtering and serialization.

These tests exercise the config layer of the composable-quantization feature and
run fast (no actual quantization / calibration):

- the ``_layer_matches_filters`` config method (class-name exact match + path regex).
- ``white_list`` / ``exclude_list`` handling inside ``get_model_info`` for the Dynamic
  and Static quant configs.
- config serialization round-trips (``to_dict`` / ``from_dict`` /
  ``from_json_string``) including the new filter attributes.
"""

import keras
import pytest

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig


@pytest.mark.parametrize("config_cls", [DynamicQuantConfig, StaticQuantConfig], ids=["dynamic_quant", "static_quant"])
class TestLayerMatchesFilters:
    """Direct unit tests for the config-level ``_layer_matches_filters`` predicate."""

    def test_no_filters_matches_everything(self, config_cls):
        assert (
            config_cls()._layer_matches_filters("net/first", "Dense") is True
        ), "Default white_list with no exclude must match every layer"

    def test_white_list_class_name_exact_match(self, config_cls):
        assert (
            config_cls(white_list=["Dense"])._layer_matches_filters("net/first", "Dense") is True
        ), "White_list matching the class name must keep the layer"

    def test_white_list_class_name_no_match(self, config_cls):
        assert (
            config_cls(white_list=["EinsumDense"])._layer_matches_filters("net/first", "Dense") is False
        ), "White_list with no matching pattern must drop the layer"

    def test_white_list_class_object_match(self, config_cls):
        # A layer class object (callable) is matched by its ``__name__``.
        assert (
            config_cls(white_list=[keras.layers.Dense])._layer_matches_filters("net/foo", "Dense") is True
        ), "White_list holding the layer class object must match by its __name__"

    def test_white_list_class_object_no_match(self, config_cls):
        assert (
            config_cls(white_list=[keras.layers.EinsumDense])._layer_matches_filters("net/first", "Dense") is False
        ), "White_list class object for a different class must drop the layer"

    def test_white_list_path_regex_match(self, config_cls):
        assert (
            config_cls(white_list=["encoder/.*"])._layer_matches_filters("net/encoder/first", "Dense") is True
        ), "White_list regex matching the path must keep the layer"

    def test_white_list_path_regex_no_match(self, config_cls):
        assert (
            config_cls(white_list=["encoder/.*"])._layer_matches_filters("net/decoder/first", "Dense") is False
        ), "White_list regex not matching the path must drop the layer"

    def test_white_list_multiple_patterns_any_match(self, config_cls):
        assert (
            config_cls(white_list=["first", "third"])._layer_matches_filters("net/third", "Dense") is True
        ), "Layer matching any white_list pattern must be kept"

    def test_exclude_class_name(self, config_cls):
        assert (
            config_cls(exclude_list=["Dense"])._layer_matches_filters("net/first", "Dense") is False
        ), "Exclude matching the class name must drop the layer"

    def test_exclude_path_regex(self, config_cls):
        assert (
            config_cls(exclude_list=["second"])._layer_matches_filters("net/second", "Dense") is False
        ), "Exclude regex matching the path must drop the layer"

    def test_exclude_no_match_keeps_layer(self, config_cls):
        assert (
            config_cls(exclude_list=["second"])._layer_matches_filters("net/first", "Dense") is True
        ), "Exclude that does not match must keep the layer"

    def test_exclude_takes_precedence_over_white_list(self, config_cls):
        # Layer matches both white_list and exclude -> excluded.
        assert (
            config_cls(white_list=["first", "second"], exclude_list=["second"])._layer_matches_filters(
                "net/second", "Dense"
            )
            is False
        ), "Exclude must win when a layer matches both white_list and exclude"

    def test_white_list_and_exclude_both_pass(self, config_cls):
        assert (
            config_cls(white_list=["first", "second"], exclude_list=["second"])._layer_matches_filters(
                "net/first", "Dense"
            )
            is True
        ), "Layer that matches white_list but not exclude must be kept"

    def test_class_name_equality_bypasses_regex(self, config_cls):
        # 'Dense' equals the class name, so it matches even though the layer id
        # does not contain the substring 'Dense'.
        assert (
            config_cls(white_list=["Dense"])._layer_matches_filters("net/foo", "Dense") is True
        ), "Exact class-name equality must match even if the path lacks the substring"

    def test_invalid_white_list_regex_raises_value_error(self, config_cls):
        with pytest.raises(ValueError):
            config_cls(white_list=["([unclosed"])._layer_matches_filters("net/first", "Dense")

    def test_invalid_exclude_regex_raises_value_error(self, config_cls):
        with pytest.raises(ValueError):
            config_cls(exclude_list=["([unclosed"])._layer_matches_filters("net/first", "Dense")


def _ids(model_info):
    return [layer_id for layer_id, _ in model_info]


def _selected_ids(cfg, model):
    """Return the layer paths a config actually selects.

    Selection is driven by ``white_list`` / ``exclude_list`` and is observed on the
    final ``to_config_mapping``.
    """
    mapping = cfg.to_config_mapping(model_info=cfg.get_model_info(model))
    return [op_name for (op_name, _op_type) in mapping]


@pytest.mark.parametrize("config_cls", [DynamicQuantConfig, StaticQuantConfig], ids=["dynamic_quant", "static_quant"])
class TestGetModelInfoFiltering:
    """White_list / exclude_list selection for both quant config types.

    The shared ``model`` fixture provides a model with three named Dense
    layers (``first``/``second``/``third``) plus an unsupported ``norm`` layer.
    Selection is resolved through ``white_list`` / ``exclude_list`` and observed on
    ``to_config_mapping``.
    """

    def test_no_filter_returns_all_dense(self, config_cls, model):
        info = config_cls().get_model_info(model)
        assert {cls_name for _, cls_name in info} == {
            "Dense"
        }, "Only Dense layers are quantizable, so all entries must be Dense"
        assert len(_selected_ids(config_cls(), model)) == 3, "Default white_list must select all three Dense layers"

    def test_unsupported_layer_type_absent(self, config_cls, model):
        info = config_cls().get_model_info(model)
        assert all(
            cls_name != "LayerNormalization" for _, cls_name in info
        ), "Unsupported LayerNormalization must never appear in model_info"

    def test_white_list_by_class_name(self, config_cls, model):
        ids = _selected_ids(config_cls(white_list=["Dense"]), model)
        assert len(ids) == 3, "White_list by class name 'Dense' must select all three Dense layers"

    def test_white_list_by_class_object(self, config_cls, model):
        # Passing the layer class object (callable) selects the same layers as its name.
        ids = _selected_ids(config_cls(white_list=[keras.layers.Dense]), model)
        assert len(ids) == 3, "White_list by class object keras.layers.Dense must select all three Dense layers"

    def test_white_list_by_path_regex_subset(self, config_cls, model):
        ids = _selected_ids(config_cls(white_list=["first"]), model)
        assert len(ids) == 1, "Only the 'first' layer must match the white_list pattern"
        assert all("first" in i for i in ids), "The single match must be the 'first' layer"

    def test_white_list_multiple_patterns(self, config_cls, model):
        ids = _selected_ids(config_cls(white_list=["first", "third"]), model)
        assert len(ids) == 2, "'first' and 'third' must match, giving two layers"
        assert all("second" not in i for i in ids), "'second' must not be selected"

    def test_exclude_by_path_regex(self, config_cls, model):
        ids = _selected_ids(config_cls(exclude_list=["second"]), model)
        assert len(ids) == 2, "Excluding 'second' must leave two layers"
        assert all("second" not in i for i in ids), "'second' must be excluded"

    def test_white_list_and_exclude_combined(self, config_cls, model):
        ids = _selected_ids(config_cls(white_list=["Dense"], exclude_list=["second"]), model)
        assert len(ids) == 2, "White_list all Dense then exclude 'second' must leave two layers"
        assert all("second" not in i for i in ids), "'second' must be excluded despite matching white_list"

    def test_white_list_no_match_returns_empty(self, config_cls, model):
        ids = _selected_ids(config_cls(white_list=["EinsumDense"]), model)
        assert ids == [], "A white_list pattern matching no supported layer must select nothing"


@pytest.mark.parametrize("config_cls", [DynamicQuantConfig, StaticQuantConfig], ids=["dynamic_quant", "static_quant"])
class TestConfigSerialization:
    """to_dict / from_dict / from_json_string round-trips with filter attributes."""

    def test_to_dict_includes_filters(self, config_cls):
        cfg = config_cls(white_list=["Dense"], exclude_list=[".*skip.*"])
        d = cfg.to_dict()
        assert d["white_list"] == ["Dense"], "to_dict must serialize the white_list filter"
        assert d["exclude_list"] == [".*skip.*"], "to_dict must serialize the exclude_list filter"

    def test_to_dict_serializes_class_object_to_name(self, config_cls):
        # Class objects in the white_list are serialized to their class name so
        # the config remains JSON-serializable and round-trips as a string.
        cfg = config_cls(white_list=[keras.layers.Dense])
        restored = config_cls.from_dict(cfg.to_dict())
        assert cfg.to_dict()["white_list"] == ["Dense"], "Class object must serialize to its __name__"
        assert restored.white_list == ["Dense"], "Class-object white_list must round-trip as its class name"

    def test_to_dict_omits_absent_filters(self, config_cls):
        d = config_cls().to_dict()
        assert "white_list" not in d, "Default white_list ('*') must be omitted from to_dict"
        assert "exclude_list" not in d, "Unset exclude_list must be omitted from to_dict"

    def test_get_params_dict_excludes_internals(self, config_cls):
        params = config_cls(white_list=["Dense"], exclude_list=[".*skip.*"]).get_params_dict()
        for internal in (
            "_exclude_list",
            "_global_config",
            "_local_config",
            "_white_list",
            "_is_initialized",
        ):
            assert internal not in params, f"Internal attribute {internal!r} must not leak into params dict"
        assert "weight_dtype" in params, "Public param 'weight_dtype' must be present"
        assert "activation_dtype" in params, "Public param 'activation_dtype' must be present"

    def test_from_dict_round_trip(self, config_cls):
        cfg = config_cls(
            weight_dtype="int8",
            activation_dtype="int8",
            white_list=["Dense"],
            exclude_list=[".*skip.*"],
        )
        restored = config_cls.from_dict(cfg.to_dict())
        assert restored.white_list == ["Dense"], "white_list must survive a to_dict/from_dict round-trip"
        assert restored.exclude_list == [".*skip.*"], "exclude_list must survive a to_dict/from_dict round-trip"
        assert restored.weight_dtype == "int8", "weight_dtype must survive the round-trip"
        assert restored.activation_dtype == "int8", "activation_dtype must survive the round-trip"

    def test_from_json_string_round_trip(self, config_cls):
        cfg = config_cls(white_list=["Dense"])
        restored = config_cls.from_json_string(cfg.to_json_string())
        assert restored.white_list == ["Dense"], "white_list must survive a JSON round-trip"
        assert restored.exclude_list is None, "Unset exclude_list must remain None after a JSON round-trip"

    def test_round_trip_without_filters(self, config_cls):
        # Regression: reconstructing a config with no filters must not raise and
        # must leave white_list at its default and exclude_list as None.
        restored = config_cls.from_dict(config_cls().to_dict())
        assert isinstance(restored, config_cls), "Round-trip must return the same config type"
        assert restored.white_list == "*", "Default white_list must remain '*' after the round-trip"
        assert restored.exclude_list is None, "Unset exclude_list must remain None after the round-trip"

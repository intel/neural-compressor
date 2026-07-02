#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for quantization config include/exclude filtering and serialization.

These tests exercise the config layer of the composable-quantization feature and
run fast (no actual quantization / calibration):

- ``_layer_matches_filter`` predicate (class-name exact match + path regex).
- ``include`` / ``exclude`` handling inside ``get_model_info`` for the Dynamic
  and Static quant configs.
- config serialization round-trips (``to_dict`` / ``from_dict`` /
  ``from_json_string``) including the new filter attributes.
"""

import pytest

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig
from neural_compressor.jax.quantization.config import _layer_matches_filter


class TestLayerMatchesFilter:
    """Direct unit tests for the low-level filter predicate."""

    def test_no_filters_includes_everything(self):
        assert _layer_matches_filter("net/first", "Dense", None, None) is True, "No filters must include every layer"

    def test_include_class_name_exact_match(self):
        assert (
            _layer_matches_filter("net/first", "Dense", ["Dense"], None) is True
        ), "Include matching the class name must keep the layer"

    def test_include_class_name_no_match(self):
        assert (
            _layer_matches_filter("net/first", "Dense", ["EinsumDense"], None) is False
        ), "Include with no matching pattern must drop the layer"

    def test_include_path_regex_match(self):
        assert (
            _layer_matches_filter("net/encoder/first", "Dense", ["encoder/.*"], None) is True
        ), "Include regex matching the path must keep the layer"

    def test_include_path_regex_no_match(self):
        assert (
            _layer_matches_filter("net/decoder/first", "Dense", ["encoder/.*"], None) is False
        ), "Include regex not matching the path must drop the layer"

    def test_include_multiple_patterns_any_match(self):
        assert (
            _layer_matches_filter("net/third", "Dense", ["first", "third"], None) is True
        ), "Layer matching any include pattern must be kept"

    def test_exclude_class_name(self):
        assert (
            _layer_matches_filter("net/first", "Dense", None, ["Dense"]) is False
        ), "Exclude matching the class name must drop the layer"

    def test_exclude_path_regex(self):
        assert (
            _layer_matches_filter("net/second", "Dense", None, ["second"]) is False
        ), "Exclude regex matching the path must drop the layer"

    def test_exclude_no_match_keeps_layer(self):
        assert (
            _layer_matches_filter("net/first", "Dense", None, ["second"]) is True
        ), "Exclude that does not match must keep the layer"

    def test_exclude_takes_precedence_over_include(self):
        # Layer matches both include and exclude -> excluded.
        assert (
            _layer_matches_filter("net/second", "Dense", ["first", "second"], ["second"]) is False
        ), "Exclude must win when a layer matches both include and exclude"

    def test_include_and_exclude_both_pass(self):
        assert (
            _layer_matches_filter("net/first", "Dense", ["first", "second"], ["second"]) is True
        ), "Layer that matches include but not exclude must be kept"

    def test_class_name_equality_bypasses_regex(self):
        # 'Dense' equals the class name, so it matches even though the layer id
        # does not contain the substring 'Dense'.
        assert (
            _layer_matches_filter("net/foo", "Dense", ["Dense"], None) is True
        ), "Exact class-name equality must match even if the path lacks the substring"

    def test_invalid_include_regex_raises_value_error(self):
        with pytest.raises(ValueError):
            _layer_matches_filter("net/first", "Dense", ["([unclosed"], None)

    def test_invalid_exclude_regex_raises_value_error(self):
        with pytest.raises(ValueError):
            _layer_matches_filter("net/first", "Dense", None, ["([unclosed"])


def _ids(model_info):
    return [layer_id for layer_id, _ in model_info]


@pytest.mark.parametrize("config_cls", [DynamicQuantConfig, StaticQuantConfig], ids=["dynamic_quant", "static_quant"])
class TestGetModelInfoFiltering:
    """Filtering behavior of ``get_model_info`` for both quant config types.

    The shared ``model`` fixture provides a model with three named Dense
    layers (``first``/``second``/``third``) plus an unsupported ``norm`` layer.
    """

    def test_no_filter_returns_all_dense(self, config_cls, model):
        info = config_cls().get_model_info(model)
        assert {cls_name for _, cls_name in info} == {
            "Dense"
        }, "Only Dense layers are quantizable, so all entries must be Dense"
        assert len(info) == 3, "The model has three Dense layers"

    def test_unsupported_layer_type_absent(self, config_cls, model):
        info = config_cls().get_model_info(model)
        assert all(
            cls_name != "LayerNormalization" for _, cls_name in info
        ), "Unsupported LayerNormalization must never appear in model_info"

    def test_include_by_class_name(self, config_cls, model):
        info = config_cls(include=["Dense"]).get_model_info(model)
        assert len(info) == 3, "Including by class name 'Dense' must match all three Dense layers"

    def test_include_by_path_regex_subset(self, config_cls, model):
        info = config_cls(include=["first"]).get_model_info(model)
        ids = _ids(info)
        assert len(ids) == 1, "Only the 'first' layer must match the include pattern"
        assert all("first" in i for i in ids), "The single match must be the 'first' layer"

    def test_include_multiple_patterns(self, config_cls, model):
        info = config_cls(include=["first", "third"]).get_model_info(model)
        ids = _ids(info)
        assert len(ids) == 2, "'first' and 'third' must match, giving two layers"
        assert all("second" not in i for i in ids), "'second' must not be included"

    def test_exclude_by_path_regex(self, config_cls, model):
        info = config_cls(exclude=["second"]).get_model_info(model)
        ids = _ids(info)
        assert len(ids) == 2, "Excluding 'second' must leave two layers"
        assert all("second" not in i for i in ids), "'second' must be excluded"

    def test_include_and_exclude_combined(self, config_cls, model):
        info = config_cls(include=["Dense"], exclude=["second"]).get_model_info(model)
        ids = _ids(info)
        assert len(ids) == 2, "Include all Dense then exclude 'second' must leave two layers"
        assert all("second" not in i for i in ids), "'second' must be excluded despite matching include"

    def test_include_no_match_returns_empty(self, config_cls, model):
        info = config_cls(include=["EinsumDense"]).get_model_info(model)
        assert info == [], "An include pattern matching no layer must yield an empty model_info"


@pytest.mark.parametrize("config_cls", [DynamicQuantConfig, StaticQuantConfig], ids=["dynamic_quant", "static_quant"])
class TestConfigSerialization:
    """to_dict / from_dict / from_json_string round-trips with filter attributes."""

    def test_to_dict_includes_filters(self, config_cls):
        cfg = config_cls(include=["Dense"], exclude=[".*skip.*"])
        d = cfg.to_dict()
        assert d["include"] == ["Dense"], "to_dict must serialize the include filter"
        assert d["exclude"] == [".*skip.*"], "to_dict must serialize the exclude filter"

    def test_to_dict_omits_absent_filters(self, config_cls):
        d = config_cls().to_dict()
        assert "include" not in d, "Unset include must be omitted from to_dict"
        assert "exclude" not in d, "Unset exclude must be omitted from to_dict"

    def test_get_params_dict_excludes_internals(self, config_cls):
        params = config_cls(include=["Dense"], exclude=[".*skip.*"]).get_params_dict()
        for internal in (
            "_include",
            "_exclude",
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
            include=["Dense"],
            exclude=[".*skip.*"],
        )
        restored = config_cls.from_dict(cfg.to_dict())
        assert restored.include == ["Dense"], "include must survive a to_dict/from_dict round-trip"
        assert restored.exclude == [".*skip.*"], "exclude must survive a to_dict/from_dict round-trip"
        assert restored.weight_dtype == "int8", "weight_dtype must survive the round-trip"
        assert restored.activation_dtype == "int8", "activation_dtype must survive the round-trip"

    def test_from_json_string_round_trip(self, config_cls):
        cfg = config_cls(include=["Dense"])
        restored = config_cls.from_json_string(cfg.to_json_string())
        assert restored.include == ["Dense"], "include must survive a JSON round-trip"
        assert restored.exclude is None, "Unset exclude must remain None after a JSON round-trip"

    def test_round_trip_without_filters(self, config_cls):
        # Regression: reconstructing a config with no filters must not raise and
        # must leave include/exclude as None.
        restored = config_cls.from_dict(config_cls().to_dict())
        assert isinstance(restored, config_cls), "Round-trip must return the same config type"
        assert restored.include is None, "Unset include must remain None after the round-trip"
        assert restored.exclude is None, "Unset exclude must remain None after the round-trip"

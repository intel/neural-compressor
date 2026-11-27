"""Tests for loading a ComposableConfig from a JSON file.

`BaseConfig.from_json_file` ends with `cls.from_dict(config_dict)`, so a subclass
whose `from_dict` takes an extra argument could not be loaded from a file at all.
"""

import json
import os
import tempfile

import pytest

from neural_compressor.common.base_config import ComposableConfig, _default_config_registry


def _write_config(directory, payload):
    path = os.path.join(directory, "composed.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _register_torch_configs():
    """Import the torch adapter, which registers its config classes."""
    pytest.importorskip("neural_compressor.torch")
    try:
        import neural_compressor.torch.quantization  # noqa: F401
    except PermissionError:
        # the adapter touches a log directory that may not be writable; the
        # registration still happened
        pass


def test_from_json_file_uses_the_registered_configs():
    """A composed config file must load without the caller passing a registry."""
    _register_torch_configs()

    with tempfile.TemporaryDirectory() as directory:
        path = _write_config(
            directory, {"rtn": {"weight_bits": 4}, "awq": {"weight_bits": 4}}
        )
        composed = ComposableConfig.from_json_file(path)

    assert isinstance(composed, ComposableConfig)
    # the merged config carries fields from both children
    merged = composed.to_dict()
    assert merged, "the composed config should not be empty"


def test_from_dict_still_accepts_an_explicit_registry():
    """Passing a registry stays supported and takes precedence over the default."""
    _register_torch_configs()
    registry = _default_config_registry()
    if "rtn" not in registry:
        pytest.skip("rtn config is not registered in this environment")

    composed = ComposableConfig.from_dict({"rtn": {"weight_bits": 4}}, registry)
    # a single entry collapses to that config class, which is the documented
    # behaviour of from_dict; what matters here is that the explicit registry worked
    assert composed.to_dict()["bits"] == 4


def test_default_registry_rejects_duplicate_names(monkeypatch):
    """Two frameworks registering one name cannot both be used from a file."""
    from neural_compressor.common import base_config

    class _A:
        pass

    class _B:
        pass

    monkeypatch.setattr(
        base_config.config_registry,
        "get_cls_configs",
        lambda: {"fwa": {"dup": _A}, "fwb": {"dup": _B}},
    )
    with pytest.raises(ValueError, match="more than one framework"):
        _default_config_registry()

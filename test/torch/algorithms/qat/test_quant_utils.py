# -*- coding: utf-8 -*-

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.qat import quant_utils
from neural_compressor.torch.algorithms.qat.quant_linear import QuantLinear
from neural_compressor.torch.algorithms.qat.tensor_quantizer import TensorQuantizer  # type: ignore


class TinyModel(nn.Module):
    """Simple hierarchical model for recursive replacement tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.block = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.lm_head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.block(x)
        return self.lm_head(x)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16)


def make_quant_cfg(
    *,
    data_type="mx_fp8",
    bits=8,
    group_size=32,
    sym=True,
    act_data_type="mx_fp8",
    act_bits=8,
    act_group_size=32,
    act_sym=True,
):
    """Build a lightweight namespace mimicking the attributes QuantLinear._setup expects."""
    return types.SimpleNamespace(
        data_type=data_type,
        bits=bits,
        group_size=group_size,
        sym=sym,
        act_data_type=act_data_type,
        act_bits=act_bits,
        act_group_size=act_group_size,
        act_sym=act_sym,
    )


@pytest.fixture
def quant_cfg():
    return make_quant_cfg()


def test_convert_replaces_class_and_calls_setup(monkeypatch, quant_cfg):
    linear = nn.Linear(4, 3)

    original_forward_id = id(QuantLinear.forward)

    quant_utils.convert(linear, quant_cfg=quant_cfg, quant_module=QuantLinear)

    assert isinstance(linear, QuantLinear)
    assert hasattr(linear.forward, "__self__") and linear.forward.__self__ is linear
    assert linear.forward.__func__ is QuantLinear.forward or id(linear.forward.__func__) == original_forward_id


def test_replace_with_quant_linear_recursive(monkeypatch, quant_cfg):
    model = TinyModel()

    quant_utils.replace_with_quant_linear(model, quant_cfg=quant_cfg)

    assert isinstance(model.fc1, QuantLinear)
    assert isinstance(model.block[0], QuantLinear)
    assert isinstance(model.block[2], QuantLinear)
    assert isinstance(model.lm_head, nn.Linear)


def test_is_quantlinear_positive_and_negative():
    q = QuantLinear()
    plain = nn.Linear(4, 2)
    assert quant_utils.is_quantlinear(q) is True
    assert quant_utils.is_quantlinear(plain) is False


def test_get_quantization_format_positive(monkeypatch):
    layer = QuantLinear()

    layer.weight_quantizer = TensorQuantizer(bits=8, data_type="mx_fp8")
    layer.weight_quantizer._disabled = False
    layer.input_quantizer = TensorQuantizer(bits=8, data_type="mx_fp8")
    layer.input_quantizer._disabled = False

    layer.weight = None
    fmt = quant_utils.get_quantization_format(layer)
    assert fmt == "MXFP8"


def test_get_quantization_format_none():
    layer = nn.Linear(4, 2)
    fmt = quant_utils.get_quantization_format(layer)
    assert fmt is None


def test_get_quantization_format_unsupported_bits_raises():
    layer = QuantLinear()
    layer.weight_quantizer = TensorQuantizer(bits=4, data_type="mx_fp8")
    layer.weight_quantizer._disabled = False
    layer.input_quantizer = TensorQuantizer(bits=4, data_type="mx_fp8")
    layer.input_quantizer._disabled = False

    with pytest.raises(NotImplementedError):
        quant_utils.get_quantization_format(layer)


def test_get_quant_config_success(monkeypatch):
    # dynamic fake module: auto_round.export.export_to_llmcompressor.config
    module_name = "auto_round.export.export_to_llmcompressor.config"

    class DummyQuantCfg:
        def __init__(self):
            self.data = {
                "provider": "dummy",
                "config_groups": {
                    "group_0": {
                        "weights": {},
                        "input_activations": {},
                    }
                },
            }

        def to_dict(self):
            return self.data

    def initialize_quantization(scheme: str):
        return DummyQuantCfg()

    # auto_round
    auto_round = types.ModuleType("auto_round")
    export = types.ModuleType("auto_round.export")
    export_to = types.ModuleType("auto_round.export.export_to_llmcompressor")
    config_mod = types.ModuleType(module_name)
    config_mod.initialize_quantization = initialize_quantization

    sys.modules["auto_round"] = auto_round
    sys.modules["auto_round.export"] = export
    sys.modules["auto_round.export.export_to_llmcompressor"] = export_to
    sys.modules[module_name] = config_mod

    cfg = quant_utils.get_quant_config(scheme="mxfp8")
    assert isinstance(cfg, dict)
    assert cfg["provider"] == "auto-round"
    assert cfg["config_groups"]["group_0"]["weights"]["is_mx"] is True
    assert cfg["config_groups"]["group_0"]["input_activations"]["is_mx"] is True


def test_convert_forward_executes(monkeypatch):
    linear = nn.Linear(5, 3)

    def fake_forward(self, x):
        return torch.zeros(x.shape[0], 3)

    monkeypatch.setattr(QuantLinear, "forward", fake_forward, raising=True)

    quant_utils.convert(linear, quant_cfg=make_quant_cfg(), quant_module=QuantLinear)
    out = linear(torch.randn(2, 5))
    assert out.shape == (2, 3)
    assert torch.all(out == 0)


def test_replace_with_quant_linear_idempotent(quant_cfg):
    model = TinyModel()
    quant_utils.replace_with_quant_linear(model, quant_cfg=quant_cfg)
    quant_utils.replace_with_quant_linear(model, quant_cfg=quant_cfg)
    assert isinstance(model.fc1, QuantLinear)


@pytest.mark.parametrize("disabled", [True, False])
def test_get_quantization_format_disabled_returns_none(disabled):
    layer = QuantLinear()
    layer.weight_quantizer = TensorQuantizer(bits=8, data_type="mx_fp8")
    layer.weight_quantizer._disabled = disabled
    layer.input_quantizer = TensorQuantizer(bits=8, data_type="mx_fp8")
    layer.input_quantizer._disabled = disabled

    fmt = quant_utils.get_quantization_format(layer)
    if disabled:
        assert fmt is None
    else:
        assert fmt == "MXFP8"

    layer.weight_quantizer = TensorQuantizer(bits=4, data_type="mx_fp4")
    layer.weight_quantizer._disabled = disabled
    layer.input_quantizer = TensorQuantizer(bits=4, data_type="mx_fp4")
    layer.input_quantizer._disabled = disabled

    fmt = quant_utils.get_quantization_format(layer)
    if disabled:
        assert fmt is None
    else:
        assert fmt == "MXFP4"

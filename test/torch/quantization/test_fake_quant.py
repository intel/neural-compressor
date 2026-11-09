# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the fake-quant mechanism (QAT/QAD shared layer)."""

import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from neural_compressor.torch.quantization import (
    FakeQuantConfig,
    QuantFormat,
    QuantizerConfig,
    QuantRule,
    TensorQuantizer,
    calibrate,
    get_format,
    prepare_qat,
    quantize,
)
from neural_compressor.torch.quantization.fake_quant import QuantLinear
from neural_compressor.torch.quantization.fake_quant.calibrator import load_amax, modelopt_megatron_translate
from neural_compressor.torch.quantization.fake_quant.ops.reference import fake_quant, fake_quant_fp8_amax
from neural_compressor.torch.quantization.fake_quant.plan import compile_plan


class TinyBlock(nn.Module):
    """Two linears with names a recipe can match."""

    def __init__(self, dim=32):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.fc2 = nn.Linear(dim * 2, dim, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.self_attn.q_proj(self.mlp.fc2(self.mlp.fc1(x)))


class TinyModel(nn.Module):
    """Minimal decoder stack with an excluded lm_head."""

    def __init__(self, dim=32, layers=1):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(dim) for _ in range(layers)])
        self.lm_head = nn.Linear(dim, 8, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


class TestFormats(unittest.TestCase):
    def test_known_formats(self):
        self.assertEqual(get_format("mxfp4").block_size, 32)
        self.assertEqual(get_format("nvfp4").block_size, 16)
        self.assertEqual(get_format("fp8_e4m3").element_bits, 8)
        self.assertEqual(get_format("mxfp8").block_size, 32)
        self.assertEqual(get_format("bf16").element_bits, 16)

    def test_custom_object_passthrough(self):
        fmt = QuantFormat(name="toy", element_bits=4)
        self.assertIs(get_format(fmt), fmt)

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            get_format("does_not_exist")

    def test_static_scheme_rejected_for_block_formats(self):
        with self.assertRaises(ValueError):
            QuantizerConfig(format="mxfp4", scheme="static")


def _ste(x, quantized):
    """Straight-through view: identical to TensorQuantizer.forward."""
    return x + (quantized - x).detach()


class TestTensorQuantizer(unittest.TestCase):
    def test_dynamic_matches_reference(self):
        torch.manual_seed(0)
        x = torch.randn(8, 64)
        for fmt in ("mxfp4", "nvfp4", "fp8_e4m3", "mxfp8"):
            quantizer = TensorQuantizer(QuantizerConfig(format=fmt))
            self.assertTrue(torch.equal(quantizer(x), _ste(x, fake_quant(x, fmt))), msg=fmt)
            self.assertIsNone(quantizer.amax)

    def test_static_fp8_matches_reference(self):
        torch.manual_seed(1)
        x = torch.randn(8, 64)
        amax = torch.tensor(7.5)
        quantizer = TensorQuantizer(QuantizerConfig(format="fp8_e4m3", scheme="static", amax=amax))
        self.assertTrue(torch.equal(quantizer(x), _ste(x, fake_quant_fp8_amax(x, amax))))
        self.assertAlmostEqual(float(quantizer.amax), 7.5)

    def test_static_without_amax_raises(self):
        quantizer = TensorQuantizer(QuantizerConfig(format="fp8_e4m3", scheme="static"))
        with self.assertRaises(RuntimeError):
            quantizer(torch.randn(4, 8))

    def test_ste_backward_is_identity(self):
        x = torch.randn(4, 32, requires_grad=True)
        quantizer = TensorQuantizer(QuantizerConfig(format="mxfp4"))
        quantizer(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_calibration_updates_amax(self):
        quantizer = TensorQuantizer(QuantizerConfig(format="fp8_e4m3", scheme="static"))
        quantizer.calibrating = True
        first = torch.full((4, 8), 2.0)
        second = torch.full((4, 8), 5.0)
        quantizer(first)
        quantizer(second)
        quantizer.finalize()
        self.assertAlmostEqual(float(quantizer.amax), 5.0)

    def test_bf16_is_noop(self):
        x = torch.randn(4, 8)
        quantizer = TensorQuantizer(QuantizerConfig(format="bf16"))
        self.assertFalse(quantizer.enabled)
        self.assertIs(quantizer(x), x)


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()

    def _rules(self):
        return [
            QuantRule(module=r".*\.mlp\.(fc1|fc2)$", weight="mxfp4", input="mxfp4"),
            QuantRule(module=r".*\.self_attn\.q_proj$", weight="fp8_e4m3"),
        ]

    def test_plan_matches_expected_modules(self):
        report = compile_plan(self.model, self._rules())
        self.assertEqual(
            set(report.plan),
            {
                "model.layers.0.mlp.fc1",
                "model.layers.0.mlp.fc2",
                "model.layers.0.self_attn.q_proj",
            },
        )
        self.assertIn("lm_head", report.excluded)

    def test_later_rule_overrides_earlier(self):
        rules = [
            QuantRule(module=r".*\.mlp\.fc1$", weight="mxfp4", input="mxfp4"),
            QuantRule(module=r".*\.mlp\.fc1$", weight="fp8_e4m3"),
        ]
        report = compile_plan(self.model, rules)
        config = report.plan["model.layers.0.mlp.fc1"]
        self.assertEqual(config.weight.format.name, "fp8_e4m3")
        self.assertEqual(config.input.format.name, "mxfp4")

    def test_unmatched_rule_raises(self):
        rules = [QuantRule(module=r".*does_not_exist$", weight="mxfp4")]
        with self.assertRaises(ValueError):
            compile_plan(self.model, rules)
        report = compile_plan(self.model, rules, strict=False)
        self.assertEqual(report.unmatched_rules, [r".*does_not_exist$"])

    def test_quantizer_level_rule(self):
        rule = QuantRule(quantizer=r".*mlp\.fc1\.weight_quantizer$", weight="mxfp4")
        report = compile_plan(self.model, [rule])
        self.assertEqual(set(report.plan), {"model.layers.0.mlp.fc1"})
        self.assertEqual(report.plan["model.layers.0.mlp.fc1"].weight.format.name, "mxfp4")
        self.assertIsNone(report.plan["model.layers.0.mlp.fc1"].input)

    def test_invalid_rule_selectors(self):
        with self.assertRaises(ValueError):
            QuantRule(weight="mxfp4")
        with self.assertRaises(ValueError):
            QuantRule(module=".*", quantizer=".*")


class TestConfigAndQuantize(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyModel()
        self.rules = [
            QuantRule(module=r".*\.mlp\.(fc1|fc2)$", weight="mxfp4", input="mxfp4"),
            QuantRule(module=r".*\.self_attn\.q_proj$", weight="fp8_e4m3"),
        ]

    def test_config_mapping_contract(self):
        cfg = FakeQuantConfig(rules=self.rules)
        mapping = cfg.to_config_mapping(model_info=cfg.get_model_info(self.model))
        self.assertTrue(mapping)
        for the_config in mapping.values():
            self.assertEqual(the_config.name, "fake_quant")

    def test_quantize_replaces_modules(self):
        model = quantize(self.model, FakeQuantConfig(rules=self.rules))
        self.assertIsInstance(model.model.layers[0].mlp.fc1, QuantLinear)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, QuantLinear)
        self.assertNotIsInstance(model.lm_head, QuantLinear)
        # Fake-quant keeps float weights: HF Trainer must still accept the model.
        self.assertFalse(getattr(model, "is_quantized", False))
        self.assertTrue(getattr(model, "is_fake_quant", False))
        # weight/bias keys are unchanged by the subclass
        self.assertIn("model.layers.0.mlp.fc1.weight", model.state_dict())

    def test_quantize_forward_and_backward(self):
        model = quantize(self.model, FakeQuantConfig(rules=self.rules))
        x = torch.randn(2, 32, requires_grad=True)
        model(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_prepare_then_calibrate(self):
        cfg = FakeQuantConfig(
            rules=[QuantRule(module=r".*\.self_attn\.q_proj$", weight=dict(format="fp8_e4m3", scheme="static"))],
        )
        model = prepare_qat(self.model, cfg)
        quantizer = model.model.layers[0].self_attn.q_proj.weight_quantizer
        self.assertTrue(quantizer.calibrating)

        def run_fn(mod, loader):
            with torch.no_grad():
                for batch in loader:
                    mod(batch)

        calibrate(model, run_fn, run_args=([torch.randn(2, 32) * 3.0],))
        self.assertIsNotNone(quantizer.amax)
        self.assertGreater(float(quantizer.amax), 0.0)

    def test_legacy_prepare_qat_mapping(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = prepare_qat(self.model, mapping={nn.Linear: "MXFP8"})
        self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in caught))
        self.assertEqual(model.model.layers[0].mlp.fc1.weight_quantizer.format.name, "mxfp8")

    def test_config_roundtrip(self):
        cfg = FakeQuantConfig(rules=self.rules, algorithm="max", strict=False)
        restored = FakeQuantConfig.from_dict(cfg.to_dict())
        self.assertEqual(len(restored.rules), len(cfg.rules))
        self.assertEqual(restored.rules[0].module, cfg.rules[0].module)
        self.assertEqual(restored.rules[0].weight.format.name, "mxfp4")

    def test_unknown_calibration_algorithm_raises(self):
        with self.assertRaises(NotImplementedError):
            FakeQuantConfig(rules=self.rules, algorithm="histogram")

    def test_needs_calibration(self):
        dynamic = FakeQuantConfig(rules=self.rules)
        self.assertFalse(dynamic.needs_calibration())
        static = FakeQuantConfig(rules=[QuantRule(module=r".*", weight=dict(format="fp8_e4m3", scheme="static"))])
        self.assertTrue(static.needs_calibration())
        with_file = FakeQuantConfig(
            rules=[QuantRule(module=r".*", weight=dict(format="fp8_e4m3", scheme="static"))],
            amax_source={"file": "somewhere.txt"},
        )
        self.assertFalse(with_file.needs_calibration())


class TestAmaxSource(unittest.TestCase):
    SUMMARY = (
        "language_model.decoder.layers.0.self_attention.linear_qkv.input_quantizer "
        "TensorQuantizer((4, 3) bit fake per-tensor amax=5.02e+01 calibrator=MaxCalibrator quant)\n"
        "language_model.decoder.layers.0.self_attention.linear_qkv.weight_quantizer "
        "TensorQuantizer((4, 3) bit fake per-tensor amax=4.36e-01 calibrator=MaxCalibrator quant)\n"
        "language_model.decoder.layers.0.mlp.linear_fc1.input_quantizer "
        "TensorQuantizer((2, 1) bit fake block_sizes={-1: 32}, amax=None calibrator=MaxCalibrator quant)\n"
    )

    def test_translate_megatron_names(self):
        self.assertEqual(
            modelopt_megatron_translate("language_model.decoder.layers.0.self_attention.linear_qkv.input_quantizer"),
            [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.k_proj",
                "layers.0.self_attn.v_proj",
            ],
        )

    def test_load_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.txt"
            path.write_text(self.SUMMARY)
            amax = load_amax(path)
        self.assertIn("layers.0.self_attn.q_proj", amax)
        self.assertAlmostEqual(float(amax["layers.0.self_attn.q_proj"]["weight"]), 0.436, places=4)
        self.assertAlmostEqual(float(amax["layers.0.self_attn.q_proj"]["input"]), 50.2, places=4)
        # MXFP4 lines have amax=None and must not appear as static values
        self.assertNotIn("layers.0.mlp.fc1", amax)

    def test_quantize_with_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.txt"
            path.write_text(self.SUMMARY)
            model = TinyModel()
            cfg = FakeQuantConfig(
                rules=[
                    QuantRule(
                        module=r".*\.self_attn\.(q|k|v)_proj$",
                        weight=dict(format="fp8_e4m3", scheme="static"),
                    ),
                    QuantRule(
                        module=r".*\.self_attn\.q_proj$",
                        input=dict(format="fp8_e4m3", scheme="static"),
                    ),
                ],
                amax_source={"file": str(path)},
            )
            model = quantize(model, cfg)
        quantizer = model.model.layers[0].self_attn.q_proj.weight_quantizer
        self.assertAlmostEqual(float(quantizer.amax), 0.436, places=4)

    def test_ambiguous_suffix_raises(self):
        from neural_compressor.torch.quantization.fake_quant.calibrator import AmaxMap

        amax = AmaxMap({
            "self_attn.q_proj": {"weight": torch.tensor(1.0)},
            "layers.0.self_attn.q_proj": {"weight": torch.tensor(2.0)},
        })
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            amax.lookup("model.layers.0.self_attn.q_proj")


if __name__ == "__main__":
    unittest.main()

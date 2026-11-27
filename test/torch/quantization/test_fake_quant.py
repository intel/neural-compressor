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
"""Unit tests for FakeQuantConfig, QuantSchemeRule, TensorQuantizer, Plan, and Calibration."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.fake_quant import load_quantizer_state_dict
from neural_compressor.torch.quantization import (
    FakeQuantConfig,
    ModuleQuantConfig,
    QuantFormat,
    QuantizerConfig,
    QuantLinear,
    QuantSchemeRule,
    TensorQuantizer,
    build_quant_plan,
    calibrate,
    calibrate_static_amax,
    convert,
    convert_to_quantized_model,
    disable_fake_quant,
    get_format,
    load_quantizer_state,
    match_quant_config,
    prepare,
    prepare_qat,
    quantize,
    save_quantizer_state,
)


class TinyHybridBlock(nn.Module):
    def __init__(self, dim: int = 32, is_linear_attn: bool = False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.down_proj = nn.Linear(dim * 2, dim, bias=False)
        if is_linear_attn:
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(dim, dim, bias=False)
            self.linear_attn.in_proj_z = nn.Linear(dim, dim, bias=False)
            self.linear_attn.in_proj_a = nn.Linear(dim, 8, bias=False)
            self.linear_attn.in_proj_b = nn.Linear(dim, 8, bias=False)
            self.linear_attn.out_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
            self.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
            self.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
            self.self_attn.o_proj = nn.Linear(dim, dim, bias=False)
        self._is_linear = is_linear_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp.down_proj(self.mlp.gate_proj(x) * self.mlp.up_proj(x))
        if self._is_linear:
            _ = self.linear_attn.in_proj_a(x), self.linear_attn.in_proj_b(x)
            h = self.linear_attn.out_proj(self.linear_attn.in_proj_qkv(h) + self.linear_attn.in_proj_z(h))
        else:
            h = self.self_attn.o_proj(self.self_attn.q_proj(h) + self.self_attn.k_proj(h) + self.self_attn.v_proj(h))
        return h


class TinyHybridModel(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyHybridBlock(dim, is_linear_attn=False), TinyHybridBlock(dim, is_linear_attn=True)]
        )
        self.lm_head = nn.Linear(dim, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


class TestFormatsAndConfigs(unittest.TestCase):
    def test_known_formats_and_capabilities(self):
        self.assertEqual(get_format("mxfp4").block_size, 32)
        self.assertTrue(get_format("mxfp4").is_block_scaled)
        self.assertFalse(get_format("mxfp4").supports_static)
        self.assertEqual(get_format("nvfp4").block_size, 16)
        self.assertTrue(get_format("nvfp4").has_global_scale)
        self.assertTrue(get_format("fp8_e4m3").supports_static)
        self.assertFalse(get_format("bf16").supports_dynamic)
        custom = QuantFormat(name="custom4", element_bits=4)
        self.assertIs(get_format(custom), custom)

    def test_reject_static_on_block_scaled_formats(self):
        for fmt in ("mxfp4", "nvfp4", "mxfp8"):
            with self.assertRaises(ValueError):
                QuantizerConfig(format=fmt, dynamic=False)

    def test_config_json_and_dict_roundtrip_preserves_exact_amax(self):
        tricky_amax = 0.4360000193119049  # exact FP32 value
        cfg = FakeQuantConfig(
            config_groups=[
                QuantSchemeRule(targets=[r"re:.*mlp\..*_proj$"], weights="mxfp4", input_activations="mxfp4"),
                QuantSchemeRule(
                    targets=[r"re:.*self_attn\..*_proj$"],
                    weights=QuantizerConfig(format="fp8_e4m3", dynamic=False, amax=tricky_amax),
                    input_activations="fp8_static",
                ),
            ],
            ignore=["lm_head"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "cfg.json"
            cfg.to_json_file(json_path)
            restored = FakeQuantConfig.from_json_file(json_path)
        self.assertEqual(len(restored.config_groups), 2)
        resolved = restored.config_groups[1].resolve()
        self.assertEqual(resolved.weights.amax, tricky_amax)
        self.assertFalse(resolved.input_activations.dynamic)

    def test_from_dict_resolves_preset_shorthand(self):
        for preset_name in (
            "mxfp4",
            "fp8_static",
            "qwen3_5_mxfp4_hybrid",
            "diffusion_mxfp4_fp8_hybrid",
        ):
            cfg = FakeQuantConfig.from_dict({"preset": preset_name, "backend": "pytorch"})
            self.assertGreater(len(cfg.config_groups), 0, msg=f"Empty config_groups for preset={preset_name}")
            self.assertEqual(cfg.backend, "pytorch")


class TestTensorQuantizerInvariants(unittest.TestCase):
    def test_fp32_amax_survives_model_to_bf16_and_fp16(self):
        model = nn.Sequential(
            QuantLinear.from_float(
                nn.Linear(32, 32, bias=False),
                ModuleQuantConfig(
                    weights=QuantizerConfig(format="fp8_e4m3", dynamic=False, amax=0.12345679104328156),
                    input_activations=QuantizerConfig(format="fp8_e4m3", dynamic=False, amax=42.87654495239258),
                ),
            )
        )
        orig_w_amax = model[0].weight_quantizer.amax.clone()
        orig_i_amax = model[0].input_quantizer.amax.clone()

        model.to(torch.bfloat16)
        self.assertEqual(model[0].weight.dtype, torch.bfloat16)
        self.assertEqual(model[0].weight_quantizer.amax.dtype, torch.float32)
        self.assertEqual(model[0].input_quantizer.amax.dtype, torch.float32)
        self.assertTrue(torch.equal(model[0].weight_quantizer.amax, orig_w_amax))
        self.assertTrue(torch.equal(model[0].input_quantizer.amax, orig_i_amax))

        model.half()
        self.assertEqual(model[0].weight_quantizer.amax.dtype, torch.float32)
        self.assertTrue(torch.equal(model[0].weight_quantizer.amax, orig_w_amax))

    def test_state_dict_loads_calibrated_amax_into_uncalibrated_quantizer(self):
        cfg = ModuleQuantConfig(weights="fp8_static", input_activations="fp8_static")
        src = QuantLinear.from_float(nn.Linear(32, 16, bias=False), cfg.clone())
        src.weight_quantizer.set_amax(1.75)
        src.input_quantizer.set_amax(9.25)

        dst = QuantLinear.from_float(nn.Linear(32, 16, bias=False), cfg.clone())
        self.assertIsNone(dst.weight_quantizer.amax)
        dst.load_state_dict(src.state_dict(), strict=True)
        self.assertEqual(dst.weight_quantizer.amax.dtype, torch.float32)
        self.assertEqual(float(dst.weight_quantizer.amax), 1.75)
        self.assertEqual(float(dst.input_quantizer.amax), 9.25)

    def test_disable_fake_quant_context_restores_exact_states(self):
        cfg = ModuleQuantConfig(weights="mxfp4", input_activations="bf16")
        qlin = QuantLinear.from_float(nn.Linear(32, 16, bias=False), cfg)
        x = torch.randn(2, 32)
        q_out = qlin(x)
        float_out = nn.functional.linear(x, qlin.weight)
        self.assertFalse(torch.equal(q_out, float_out))

        with disable_fake_quant(qlin):
            self.assertFalse(qlin.weight_quantizer.is_enabled)
            self.assertTrue(torch.equal(qlin(x), float_out))

        self.assertTrue(qlin.weight_quantizer.is_enabled)
        self.assertFalse(qlin.input_quantizer.is_enabled)


class TestPlanCalibrationAndQwen35Preset(unittest.TestCase):
    def test_qwen35_hybrid_preset_routing_and_calibration(self):
        torch.manual_seed(0)
        model = TinyHybridModel(dim=32)
        cfg = FakeQuantConfig.qwen3_5_mxfp4_hybrid()
        report = build_quant_plan(model, cfg)

        # 2 layers * 3 MLP = 6 MXFP4; 1 self_attn * 4 + 1 linear_attn * 3 (qkv, z, out) = 7 FP8 static -> 13 total
        self.assertEqual(len(report.plan), 13)
        self.assertIn("lm_head", report.ignored)
        self.assertIn("model.layers.1.linear_attn.in_proj_a", report.ignored)
        self.assertIn("model.layers.1.linear_attn.in_proj_b", report.ignored)
        self.assertEqual(report.plan["model.layers.0.mlp.gate_proj"].weights.format.name, "mxfp4")
        self.assertEqual(report.plan["model.layers.1.linear_attn.in_proj_z"].weights.format.name, "fp8_e4m3")
        self.assertFalse(report.plan["model.layers.1.linear_attn.in_proj_z"].weights.dynamic)

        prepare(model, cfg)
        summary = calibrate_static_amax(model, dataloader=[torch.randn(2, 32) * 2.5, torch.randn(2, 32) * 4.0])
        self.assertEqual(len(summary), 14)  # 7 FP8 modules * 2 (weight + input)
        convert(model)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "quantizer_state.json"
            save_quantizer_state(model, state_path)

            fresh = TinyHybridModel(dim=32)
            fresh.load_state_dict(model.state_dict(), strict=False)
            cfg_with_state = FakeQuantConfig.qwen3_5_mxfp4_hybrid(quantizer_state_path=state_path)
            quantize(fresh, cfg_with_state)

            x = torch.randn(2, 32)
            torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)

            # Verify prepare_qat with pre-calibrated state and load_quantizer_state API
            fresh2 = TinyHybridModel(dim=32)
            fresh2.load_state_dict(model.state_dict(), strict=False)
            prepare_qat(fresh2, cfg)
            self.assertTrue(fresh2.model.layers[0].self_attn.q_proj.input_quantizer.is_calibrating)
            load_result = load_quantizer_state(fresh2, state_path, strict=True)
            self.assertEqual(len(load_result["loaded"]), 14)
            self.assertFalse(fresh2.model.layers[0].self_attn.q_proj.input_quantizer.is_calibrating)
            torch.testing.assert_close(fresh2(x), model(x), rtol=0, atol=0)

            fresh3 = TinyHybridModel(dim=32)
            fresh3.load_state_dict(model.state_dict(), strict=False)
            prepare_qat(fresh3, cfg_with_state)
            self.assertFalse(fresh3.model.layers[0].self_attn.q_proj.input_quantizer.is_calibrating)
            torch.testing.assert_close(fresh3(x), model(x), rtol=0, atol=0)

    def test_legacy_prepare_qat_mapping_and_inc_dict_quantize(self):
        model = TinyHybridModel(dim=32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare_qat(model, mapping={nn.Linear: "MXFP8"})
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertIsInstance(model.model.layers[0].mlp.gate_proj, QuantLinear)
        self.assertNotIsInstance(model.lm_head, QuantLinear)

        model2 = TinyHybridModel(dim=32)
        cfg = FakeQuantConfig.mxfp4()
        quantize(model2, {"fake_quant": cfg.to_dict()})
        self.assertTrue(getattr(model2, "is_fake_quant", False))
        self.assertFalse(getattr(model2, "is_quantized", False))

    def test_calibrate_inc_run_fn_entry(self):
        model = TinyHybridModel(dim=32)
        cfg = FakeQuantConfig.fp8_static()
        quantize(model, cfg, run_fn=lambda m, batch: m(batch), run_args=(torch.randn(2, 32),))
        q = model.model.layers[0].self_attn.q_proj.input_quantizer
        self.assertIsNotNone(q.amax)
        self.assertFalse(q.is_calibrating)

    def test_convert_modelopt_amax_tool(self):
        tool_path = Path(__file__).resolve().parents[3] / "examples/pytorch/llm/qwen/qad/tools/convert_modelopt_amax.py"
        spec = importlib.util.spec_from_file_location("convert_modelopt_amax", tool_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        txt = (
            "language_model.decoder.layers.0.self_attention.linear_qkv.input_quantizer "
            "TensorQuantizer((4, 3) bit fake per-tensor amax=5.02e+01 calibrator=MaxCalibrator quant)\n"
            "language_model.decoder.layers.0.self_attention.linear_qkv.weight_quantizer "
            "TensorQuantizer((4, 3) bit fake per-tensor amax=4.36e-01 calibrator=MaxCalibrator quant)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "amax.txt"
            dst = Path(tmp) / "state.json"
            src.write_text(txt)
            payload = mod.convert_modelopt_amax(src, dst, hf_prefixes=("model.layers",))
            loaded_json = json.loads(dst.read_text())
        self.assertEqual(payload, loaded_json)
        qmap = payload["quantizers"]
        self.assertAlmostEqual(
            float.fromhex(qmap["model.layers.0.self_attn.q_proj.weight_quantizer"]["amax_hex"]),
            0.436,
            places=6,
        )
        self.assertAlmostEqual(
            float.fromhex(qmap["model.layers.0.linear_attn.in_proj_z.input_quantizer"]["amax_hex"]),
            50.2,
            places=5,
        )

    def test_calibrate_bypasses_dynamic_quantizers_and_strips_checkpoint_wrappers(self):
        torch.manual_seed(0)
        float_model = TinyHybridModel(dim=32).eval()
        x = torch.randn(4, 32)
        seen_float_inputs: list[torch.Tensor] = []
        h = float_model.model.layers[1].linear_attn.in_proj_qkv.register_forward_pre_hook(
            lambda _m, inp: seen_float_inputs.append(inp[0].detach().clone())
        )
        with torch.no_grad():
            float_model(x)
        h.remove()
        expected_amax = float(seen_float_inputs[0].abs().amax().to(torch.float32))

        q_model = TinyHybridModel(dim=32)
        q_model.load_state_dict(float_model.state_dict())
        prepare_qat(q_model, FakeQuantConfig.qwen3_5_mxfp4_hybrid())
        calibrate_static_amax(q_model, dataloader=[x])

        q_in = q_model.model.layers[1].linear_attn.in_proj_qkv.input_quantizer
        self.assertEqual(float(q_in.amax), expected_amax)
        self.assertTrue(q_model.model.layers[0].mlp.gate_proj.weight_quantizer.is_enabled)

        wrapped_state = {
            f"module.model.layers.1._checkpoint_wrapped_module._fsdp_wrapped_module."
            f"linear_attn.in_proj_qkv.input_quantizer": {"amax_hex": expected_amax.hex()}
        }
        fresh = TinyHybridModel(dim=32)
        prepare_qat(fresh, FakeQuantConfig.qwen3_5_mxfp4_hybrid())
        res = load_quantizer_state_dict(fresh, wrapped_state, strict=False)
        self.assertIn("model.layers.1.linear_attn.in_proj_qkv.input_quantizer", res["loaded"])
        self.assertEqual(
            float(fresh.model.layers[1].linear_attn.in_proj_qkv.input_quantizer.amax),
            expected_amax,
        )

        from neural_compressor.torch.algorithms.fake_quant import (
            compile_plan_from_info,
            match_quant_config_from_info,
        )
        from neural_compressor.torch.quantization.config import FakeQuantConfig as ConfigFacadeFakeQuantConfig

        self.assertIs(ConfigFacadeFakeQuantConfig, FakeQuantConfig)
        self.assertIs(compile_plan_from_info, match_quant_config_from_info)
        with tempfile.TemporaryDirectory() as tmp:
            qstate_file = Path(tmp) / "quantizer_state.json"
            qstate_file.write_text(json.dumps({"quantizers": wrapped_state}))
            cfg_with_state = FakeQuantConfig.qwen3_5_mxfp4_hybrid()
            cfg_with_state.quantizer_state_path = str(qstate_file)
            report = compile_plan_from_info(
                [("model.layers.1.linear_attn.in_proj_qkv", "Linear")],
                cfg_with_state,
            )
            self.assertEqual(
                report.plan["model.layers.1.linear_attn.in_proj_qkv"].input_activations.amax,
                expected_amax,
            )

    def test_idiomatic_conversion_api_and_flat_quantization_facade(self):
        import subprocess
        import sys

        import neural_compressor.torch.quantization as quant_pkg

        quant_dir = Path(quant_pkg.__file__).resolve().parent
        subpackages = [p.name for p in quant_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
        self.assertEqual(subpackages, [], f"Expected flat quantization facade, found {subpackages}")

        model = TinyHybridModel(dim=32)
        cfg = FakeQuantConfig.mxfp4()
        matched = match_quant_config(model, cfg)
        self.assertIs(matched.module_configs, matched.plan)
        self.assertGreater(len(matched.module_configs), 0)

        converted = convert_to_quantized_model(model, cfg)
        self.assertEqual(set(converted.module_configs.keys()), set(matched.module_configs.keys()))
        self.assertIsInstance(model.model.layers[0].mlp.gate_proj, QuantLinear)

        for stmt in (
            "import neural_compressor.torch.algorithms.fake_quant, neural_compressor.torch.quantization",
            "import neural_compressor.torch.quantization, neural_compressor.torch.algorithms.fake_quant",
            "import importlib; at = importlib.import_module('neural_compressor.torch.quantization.autotune'); assert callable(at.quantize)",
        ):
            proc = subprocess.run(
                [sys.executable, "-c", stmt],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()

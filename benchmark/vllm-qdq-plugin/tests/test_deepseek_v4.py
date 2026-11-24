# SPDX-License-Identifier: Apache-2.0
"""CPU regressions; real vLLM check is opt-in via VLLM_QDQ_TEST_DEEPSEEK_V4=1."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

SOURCE = Path(__file__).resolve().parents[1] / "src/vllm_qdq_plugin/deepseek_v4.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("_deepseek_v4_plugin_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepSeekV4Tests(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin()

    def layer(self):
        layer = torch.nn.Module()
        layer.prefix = "model.layers.0.attn.wo_a"
        layer.is_bmm, layer.bmm_batch_size = True, 2
        layer.weight = torch.nn.Parameter(
            torch.arange(-64, 64).float().reshape(4, 32).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(torch.tensor([[125], [126], [127], [128]], dtype=torch.uint8), False)
        return layer

    def test_chunking_and_grouped_bmm(self):
        layer = self.layer()
        expected = (layer.weight.float() * torch.tensor([[0.25], [0.5], [1.0], [2.0]])).bfloat16()
        for chunk in (1, 3, 256):
            with self.subTest(chunk=chunk):
                actual = self.plugin._dequantize_mxfp8(layer.weight, layer.weight_scale, chunk)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.plugin._prepare_wo_a(layer)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(layer.weight.requires_grad)
        self.assertFalse(hasattr(layer, "weight_scale"))
        x = torch.ones(3, 2, 32, dtype=torch.bfloat16)
        actual = torch.bmm(x.transpose(0, 1), layer.weight.view(2, 2, 32).transpose(1, 2)).transpose(0, 1)
        expected = torch.stack([x[:, g] @ expected[g * 2 : (g + 1) * 2].T for g in range(2)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_e8m0_zero_byte(self):
        weight = torch.full((1, 32), 128.0).to(torch.float8_e4m3fn)
        actual = self.plugin._dequantize_mxfp8(weight, torch.zeros((1, 1), dtype=torch.uint8))
        self.assertEqual(actual[0, 0].item(), 2.0**-120)

    def test_invalid_layout_and_nonfinite_fail_without_mutation(self):
        for kind in ("weight_dtype", "scale_dtype", "scale_shape", "groups", "nan_scale", "overflow"):
            layer = self.layer()
            if kind == "weight_dtype":
                layer.weight = torch.nn.Parameter(layer.weight.float(), False)
            elif kind == "scale_dtype":
                layer.weight_scale = torch.nn.Parameter(layer.weight_scale.float(), False)
            elif kind == "scale_shape":
                layer.weight_scale = torch.nn.Parameter(torch.ones((4, 2), dtype=torch.uint8), False)
            elif kind == "groups":
                layer.bmm_batch_size = 3
            else:
                layer.weight_scale.data.fill_(255 if kind == "nan_scale" else 254)
            old_weight, old_scale = layer.weight, layer.weight_scale
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.plugin._prepare_wo_a(layer)
            self.assertIs(layer.weight, old_weight)
            self.assertIs(layer.weight_scale, old_scale)

    def test_default_off_does_not_import_optional_runtime(self):
        original_import = __import__

        def guarded(name, *args, **kwargs):
            if name.startswith(("torch", "vllm")):
                self.fail(f"Unexpected optional import: {name}")
            return original_import(name, *args, **kwargs)

        for value in ("0", ""):
            with (
                mock.patch.dict(os.environ, {"VLLM_DEEPSEEK_V4_WO_A_BF16": value}),
                mock.patch("builtins.__import__", side_effect=guarded),
            ):
                load_plugin().register()

    def runtime(self, version="0.33.0"):
        original_call = mock.Mock(return_value="original-result")

        class Scheme:
            def process_weights_after_loading(self, layer):
                return original_call(self, layer)

        Scheme.original_call = original_call

        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4")))
        get_config = mock.Mock(return_value=config)
        modules = {}
        for name, attrs in (
            ("vllm", {"__version__": version}),
            ("vllm.config", {"get_current_vllm_config": get_config}),
            ("vllm.logger", {"init_logger": mock.Mock()}),
            (
                "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8",
                {"CompressedTensorsW8A8Mxfp8": Scheme},
            ),
        ):
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)
        return Scheme, config, get_config, modules

    @mock.patch.dict(os.environ, {"VLLM_DEEPSEEK_V4_WO_A_BF16": "1", "VLLM_QDQ": "0"})
    def test_routing_is_narrow_and_registration_idempotent(self):
        scheme, config, get_config, modules = self.runtime()
        with mock.patch.dict(sys.modules, modules):
            self.plugin.register()
            patched = scheme.process_weights_after_loading
            load_plugin().register()
            self.assertIs(scheme.process_weights_after_loading, patched)
            for kind in ("other_model", "ordinary_linear", "other_prefix", "missing_attributes"):
                layer = self.layer()
                config.model_config.hf_config.model_type = "other" if kind == "other_model" else "deepseek_v4"
                if kind == "ordinary_linear":
                    layer.is_bmm = False
                elif kind == "other_prefix":
                    layer.prefix = "model.layers.0.attn.wo_b"
                elif kind == "missing_attributes":
                    del layer.is_bmm
                    del layer.prefix
                old_weight, old_scale = layer.weight, layer.weight_scale
                with self.subTest(kind=kind):
                    self.assertEqual(scheme().process_weights_after_loading(layer), "original-result")
                    self.assertIs(layer.weight, old_weight)
                    self.assertIs(layer.weight_scale, old_scale)
            self.assertEqual(scheme.original_call.call_count, 4)
            self.assertEqual(get_config.call_count, 1)
            config.model_config.hf_config.model_type = "deepseek_v4"
            layer = self.layer()
            scheme().process_weights_after_loading(layer)
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertEqual(scheme.original_call.call_count, 4)

    @mock.patch.dict(os.environ, {"VLLM_DEEPSEEK_V4_WO_A_BF16": "1"})
    def test_unsupported_version_fails_before_patching(self):
        scheme, _, _, modules = self.runtime(version="0.34.0")
        original = scheme.process_weights_after_loading
        with mock.patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "0.33.0"):
            self.plugin.register()
        self.assertIs(scheme.process_weights_after_loading, original)

    @unittest.skipUnless(os.getenv("VLLM_QDQ_TEST_DEEPSEEK_V4") == "1", "requires vLLM 0.33.0")
    def test_real_scheme_and_native_bf16_o_proj(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import (
            CompressedTensorsW8A8Mxfp8,
        )
        from vllm.models.deepseek_v4.nvidia.ops import o_proj

        cls = CompressedTensorsW8A8Mxfp8
        original = cls.process_weights_after_loading
        layer = self.layer()
        expected_weight = (layer.weight.float() * torch.tensor([[0.25], [0.5], [1.0], [2.0]])).bfloat16()
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4")))
        try:
            with (
                mock.patch.dict(os.environ, {"VLLM_DEEPSEEK_V4_WO_A_BF16": "1"}),
                mock.patch("vllm.config.get_current_vllm_config", return_value=config),
            ):
                self.plugin.register()
                cls().process_weights_after_loading(layer)
            x = torch.ones(3, 2, 32, dtype=torch.bfloat16)
            # Stub only CUDA inverse RoPE; execute native dtype dispatch and bmm.
            with (
                mock.patch.object(o_proj, "fused_inv_rope_fp8_quant", return_value=(x, None)) as inv_rope,
                mock.patch.object(o_proj, "fp8_einsum", side_effect=AssertionError("FP8 path used")),
            ):
                actual = o_proj.deep_gemm_fp8_o_proj(
                    x,
                    torch.arange(3),
                    torch.empty(0),
                    layer,
                    lambda z: z,
                    n_groups=2,
                    heads_per_group=1,
                    nope_dim=32,
                    rope_dim=0,
                    o_lora_rank=2,
                    einsum_recipe=(1, 128, 128),
                    tma_aligned_scales=False,
                )
            self.assertFalse(inv_rope.call_args.kwargs["quantize"])
            expected = expected_weight.sum(dim=1).expand(3, -1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            cls.process_weights_after_loading = original


if __name__ == "__main__":
    unittest.main()

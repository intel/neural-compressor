# SPDX-License-Identifier: Apache-2.0
"""Opt-in real vLLM 0.33 scheme/native BF16 branch check; no GPU allocation."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


@unittest.skipUnless(os.getenv("VLLM_QDQ_TEST_DEEPSEEK_V4") == "1", "requires vLLM 0.33.0")
class DeepSeekV4IntegrationTests(unittest.TestCase):
    def test_real_scheme_and_native_bf16_o_proj(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import (
            CompressedTensorsW8A8Mxfp8,
        )
        from vllm.models.deepseek_v4.nvidia.ops import o_proj

        source = Path(__file__).resolve().parents[1] / "src/vllm_qdq_plugin/deepseek_v4.py"
        spec = importlib.util.spec_from_file_location("_deepseek_v4_integration", source)
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        cls = CompressedTensorsW8A8Mxfp8
        original = cls.process_weights_after_loading
        layer = torch.nn.Module()
        layer.prefix, layer.is_bmm, layer.bmm_batch_size = "model.layers.0.attn.wo_a", True, 2
        layer.weight = torch.nn.Parameter(torch.ones(4, 32).to(torch.float8_e4m3fn), False)
        layer.weight_scale = torch.nn.Parameter(torch.full((4, 1), 127, dtype=torch.uint8), False)
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4")))
        try:
            with (
                mock.patch.dict(os.environ, {"VLLM_DEEPSEEK_V4_WO_A_BF16": "1"}),
                mock.patch("vllm.config.get_current_vllm_config", return_value=config),
            ):
                plugin.register()
                # Exercise the actual vLLM class, not a replacement scheme.
                cls().process_weights_after_loading(layer)
            x = torch.ones(3, 2, 32, dtype=torch.bfloat16)
            # Only the CUDA inverse-RoPE helper is stubbed. Execute vLLM's real
            # dtype dispatch, grouped reshape, BF16 bmm, and output projection.
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
            torch.testing.assert_close(actual, torch.full((3, 4), 32, dtype=torch.bfloat16))
        finally:
            cls.process_weights_after_loading = original


if __name__ == "__main__":
    unittest.main()

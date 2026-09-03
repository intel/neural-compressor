import json
import unittest

import torch
from nvfp4_hw.inc_nvfp4_ue5m3_scheme import INCNvfp4UE5M3Scheme
from nvfp4_hw.patch import apply_patches
from nvfp4_hw.xkernels_ops import mxfp4_ue5m3_bf16_gemm
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import factory
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import nvfp4_e5m3_qdq


class _FusedMoE:
    pass


class Nvfp4UE5M3ConfigTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_xkernels_gemm_supports_dynamo_fullgraph_capture(self) -> None:
        from xkernels import prepare_mxfp4_ue5m3_weight

        weight = torch.randint(0, 256, (64, 64), device="cuda", dtype=torch.uint8)
        scales = torch.randint(0, 255, (64, 8), device="cuda", dtype=torch.uint8)
        prepared = prepare_mxfp4_ue5m3_weight(weight, scales, 16)
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        expected = mxfp4_ue5m3_bf16_gemm(x, prepared)
        compiled = torch.compile(
            lambda value: mxfp4_ue5m3_bf16_gemm(value, prepared),
            backend="eager",
            fullgraph=True,
        )

        self.assertTrue(torch.equal(compiled(x), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qdq_supports_dynamo_fullgraph_capture(self) -> None:
        x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
        expected = nvfp4_e5m3_qdq(x)
        compiled = torch.compile(nvfp4_e5m3_qdq, backend="eager", fullgraph=True)

        self.assertTrue(torch.equal(compiled(x), expected))

    def test_global_nvfp4_e5m3_config_selects_linear_method(self) -> None:
        apply_patches()
        config = INCConfig.from_config(
            {
                "bits": 4,
                "group_size": 16,
                "sym": True,
                "data_type": "nvfp4_v2",
                "packing_format": "auto_round:llm_compressor_nvfp4_e5m3",
            }
        )

        layer_config = config.config_parser.resolve(object(), "model.layers.0.self_attn.q_proj")
        scheme = factory.resolve_scheme(layer_config)
        method = scheme.get_linear_method(config, object(), "model.layers.0.self_attn.q_proj", layer_config)

        self.assertEqual(layer_config.data_type, "nvfp4_v2")
        self.assertEqual(layer_config.group_size, 16)
        self.assertIsInstance(scheme, INCNvfp4UE5M3Scheme)
        self.assertIsInstance(method, INCLinearMethod)
        self.assertEqual(method.scheme.__class__.__name__, "INCNvfp4UE5M3LinearMethod")

    def test_extra_config_data_type_selects_ue5m3_scheme(self) -> None:
        apply_patches()
        config = INCConfig.from_config(
            {
                "bits": 8,
                "group_size": 32,
                "sym": True,
                "data_type": "mx_fp",
                "packing_format": "auto_round:llm_compressor",
                "extra_config": {
                    "mlp.experts": {
                        "bits": 4,
                        "group_size": 16,
                        "data_type": "nvfp4_v2",
                    }
                },
            }
        )

        layer_config = config.config_parser.resolve(_FusedMoE(), "model.language_model.layers.0.mlp.experts")

        self.assertEqual(layer_config.data_type, "nvfp4_v2")
        self.assertEqual(layer_config.group_size, 16)
        self.assertIsInstance(factory.resolve_scheme(layer_config), INCNvfp4UE5M3Scheme)

    def test_regular_layer_keeps_top_level_data_type(self) -> None:
        apply_patches()
        config = INCConfig.from_config(
            {
                "bits": 8,
                "group_size": 32,
                "sym": True,
                "data_type": "mx_fp",
                "packing_format": "auto_round:llm_compressor",
                "extra_config": {"mlp.experts": {"data_type": "nvfp4_v2"}},
            }
        )

        layer_config = config.config_parser.resolve(object(), "model.layers.0.self_attn.q_proj")

        self.assertEqual(layer_config.data_type, "mx_fp")

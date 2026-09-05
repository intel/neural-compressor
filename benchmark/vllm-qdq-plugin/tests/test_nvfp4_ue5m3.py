import json
import unittest
from unittest import mock

import torch
from nvfp4_hw.inc_nvfp4_ue5m3_linear import INCNvfp4UE5M3LinearMethod
from nvfp4_hw.inc_nvfp4_ue5m3_scheme import INCNvfp4UE5M3Scheme
from nvfp4_hw.patch import apply_patches
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import factory
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter
from vllm_qdq_plugin.qdq.nvfp4_e5m3 import (
    _nvfp4_e5m3_qdq_reference,
    decode_ue5m3,
    nvfp4_e5m3_qdq,
)
from vllm_qdq_plugin.qdq.nvfp4_e5m3_cute import nvfp4_e5m3_weight_dequant_cute


class _FusedMoE:
    pass


def _decode_e2m1(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.int32)
    sign = 1.0 - 2.0 * ((value >> 3) & 1).float()
    exponent = (value >> 1) & 0x3
    mantissa = value & 1
    subnormal = mantissa.float() * 0.5
    normal = torch.ldexp(1.0 + mantissa.float() * 0.5, exponent - 1)
    return torch.where(exponent == 0, subnormal, normal) * sign


def _dequantize_weight_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    low = _decode_e2m1(packed & 0xF)
    high = _decode_e2m1((packed >> 4) & 0xF)
    values = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    decoded_scales = decode_ue5m3(scales).repeat_interleave(group_size, dim=1)
    return (values * decoded_scales).to(torch.bfloat16)


class Nvfp4UE5M3ConfigTests(unittest.TestCase):
    @staticmethod
    def _make_linear_method() -> INCNvfp4UE5M3LinearMethod:
        method = object.__new__(INCNvfp4UE5M3LinearMethod)
        method.group_size = 16
        return method

    @staticmethod
    def _make_dense_layer(device: torch.device) -> torch.nn.Module:
        layer = torch.nn.Module()
        with (
            mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
            mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
        ):
            layer.weight_packed = ModelWeightParameter(
                data=torch.randint(0, 256, (64, 32), device=device, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=None,
            )
            layer.weight_scale = GroupQuantScaleParameter(
                data=torch.randint(0x40, 0x79, (64, 4), device=device, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=None,
            )
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 64
        layer.params_dtype = torch.bfloat16
        return layer

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_marlin_dense_matches_dequantized_reference(self) -> None:
        torch.manual_seed(1)
        method = self._make_linear_method()
        layer = self._make_dense_layer(torch.device("cuda"))
        packed = layer.weight_packed.detach().clone()
        scales = layer.weight_scale.detach().clone()
        x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(
            nvfp4_e5m3_qdq(x, 16),
            _dequantize_weight_reference(packed, scales, 16),
        )

        method.process_weights_after_loading(layer)
        actual = method.apply_weights(layer, x)

        self.assertEqual(layer.weight.dtype, torch.int32)
        self.assertFalse(hasattr(layer, "weight_packed"))
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cute_weight_dequant_matches_reference(self) -> None:
        weight = torch.randint(0, 256, (64, 64), device="cuda", dtype=torch.uint8)
        scales = torch.randint(0, 255, (64, 8), device="cuda", dtype=torch.uint8)
        expected = _dequantize_weight_reference(weight, scales, 16)
        actual = nvfp4_e5m3_weight_dequant_cute(weight, scales, 16)
        compiled = torch.compile(
            lambda packed, scale: nvfp4_e5m3_weight_dequant_cute(packed, scale, 16),
            backend="eager",
            fullgraph=True,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(compiled(weight, scales), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qdq_supports_dynamo_fullgraph_capture(self) -> None:
        x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
        expected = _nvfp4_e5m3_qdq_reference(x, 16)
        compiled = torch.compile(nvfp4_e5m3_qdq, backend="eager", fullgraph=True)

        self.assertTrue(torch.equal(nvfp4_e5m3_qdq(x), expected))
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

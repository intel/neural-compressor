import json
import unittest

from nvfp4_hw.inc_nvfp4_ue5m3_scheme import INCNvfp4UE5M3Scheme
from nvfp4_hw.patch import apply_patches
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import factory


class _FusedMoE:
    pass


class Nvfp4UE5M3ConfigTests(unittest.TestCase):
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

"""Opt-in real vLLM/Omni CPU configuration tests (no model weights loaded)."""

import os
import unittest


@unittest.skipUnless(os.environ.get("VLLM_QDQ_TEST_OMNI") == "1", "requires installed vLLM and Omni")
class OmniINCIntegrationTests(unittest.TestCase):
    def test_real_config_dispatch_with_qdq_disabled(self):
        import torch
        from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
        from vllm.model_executor.layers.quantization.inc import INCConfig
        from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
        from vllm_omni.quantization.inc_config import OmniINCConfig
        from vllm_qdq_plugin.quantization.inc_nvfp4_e5m3_linear import INCNvfp4UE5M3LinearMethod
        from vllm_qdq_plugin.quantization.omni import register

        base_method = INCConfig.get_quant_method
        self.assertEqual(os.environ.get("VLLM_QDQ"), "0")
        register()
        first = OmniINCConfig.get_quant_method
        register()
        self.assertIs(OmniINCConfig.get_quant_method, first)
        self.assertIs(INCConfig.get_quant_method, base_method)

        # get_quant_method only needs the layer type; do not allocate weights.
        layer = object.__new__(ReplicatedLinear)
        torch.nn.Module.__init__(layer)
        for dtype, packing in (
            ("nvfp4_v2", "auto_round:llm_compressor_nvfp4_e5m3"),
            ("nvfp4_v2_gs", "auto_round:llm_compressor_nvfp4_e5m3_gs"),
        ):
            config = OmniINCConfig.from_config(
                dict(
                    bits=4,
                    group_size=16,
                    sym=True,
                    data_type=dtype,
                    packing_format=packing,
                    block_name_to_quantize="blocks",
                    extra_config={"proj_out": {"bits": 16, "act_bits": 16}},
                )
            )
            config.apply_prefix_map({"blocks": "transformer.blocks", "proj_out": "transformer.proj_out"})
            method = config.get_quant_method(layer, "transformer.blocks.0.attn1.to_q")
            self.assertIsInstance(method, INCLinearMethod)
            self.assertIsInstance(method.scheme, INCNvfp4UE5M3LinearMethod)
            self.assertIsInstance(config.get_quant_method(layer, "transformer.proj_out"), UnquantizedLinearMethod)

        # Plain NVFP4 must still use Omni's compressed-tensors implementation.
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsLinearMethod,
        )

        config = OmniINCConfig.from_config(
            dict(
                bits=4,
                group_size=16,
                sym=True,
                data_type="nv_fp",
                packing_format="auto_round:llm_compressor",
            )
        )
        self.assertIsInstance(config.get_quant_method(layer, "blocks.0.to_q"), CompressedTensorsLinearMethod)


if __name__ == "__main__":
    unittest.main()

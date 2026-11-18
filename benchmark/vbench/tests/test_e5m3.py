"""CPU config checks: distinguish E5M3 packed exports from native NVFP4."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5m3_benchmark_test", Path(__file__).resolve().parents[1] / "scripts/benchmark.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class E5M3ConfigTests(unittest.TestCase):
    def test_strict_e5m3_config_for_both_transformers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))
            quant = dict(
                quant_method="auto-round",
                bits=4,
                act_bits=4,
                group_size=16,
                act_group_size=16,
                data_type="nvfp4_v2",
                act_data_type="nvfp4_v2",
                sym=True,
                act_sym=True,
                packing_format="auto_round:llm_compressor_nvfp4_e5m3",
                act_dynamic=True,
            )
            for name in ("transformer", "transformer_2"):
                component = root / name
                component.mkdir()
                (component / "config.json").write_text(json.dumps({"_class_name": "WanTransformer3DModel"}))
                (component / "quantization_config.json").write_text(json.dumps(quant))
                (component / "test.safetensors").write_bytes(b"config-only fixture")
            model = b.validate_model(root, "t2v", "nvfp4_e5m3")
            self.assertEqual(model["precision"], "nvfp4_e5m3")
            with self.assertRaises(RuntimeError):
                b.validate_model(root, "t2v", "nvfp4")
            quant["act_dynamic"] = False
            (root / "transformer_2/quantization_config.json").write_text(json.dumps(quant))
            with self.assertRaisesRegex(RuntimeError, "dynamic activations"):
                b.validate_model(root, "t2v", "nvfp4_e5m3")


if __name__ == "__main__":
    unittest.main()

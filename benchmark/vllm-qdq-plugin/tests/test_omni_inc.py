"""CPU-only compatibility regressions; no torch/vLLM/Omni installation needed."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock


PLUGIN = Path(__file__).resolve().parents[1]


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "_omni_inc_test.omni", PLUGIN / "src/vllm_qdq_plugin/quantization/omni.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OmniINCTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()
        self.base_call = mock.Mock(return_value="inc-method")
        self.omni_call = mock.Mock(return_value="omni-method")
        base_call, omni_call = self.base_call, self.omni_call

        class INCConfig:
            SUPPORTED_DTYPES = {"int", "mx_fp", "nv_fp"}
            SUPPORTED_FORMATS = {"auto_round:llm_compressor"}

            def get_quant_method(self, layer, prefix):
                return base_call(self, layer, prefix)

        class OmniINCConfig(INCConfig):
            def get_quant_method(self, layer, prefix):
                return omni_call(self, layer, prefix)

        self.base, self.omni = INCConfig, OmniINCConfig
        self.register_inc = mock.Mock()
        modules = {}
        for name, attributes in (
            ("vllm_omni.quantization.inc_config", {"OmniINCConfig": self.omni}),
            ("vllm.model_executor.layers.quantization.inc", {"INCConfig": self.base}),
            ("_omni_inc_test.patch", {"register": self.register_inc}),
        ):
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        self.modules = modules

    def test_import_does_not_load_optional_packages(self):
        original_import = __import__

        def guarded(name, *args, **kwargs):
            if name.startswith("vllm"):
                self.fail(f"Unexpected import at module load: {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            load_adapter()

    def test_e5m3_routes_to_inc_and_other_types_keep_omni_dispatch(self):
        original_base = self.base.get_quant_method
        with mock.patch.dict(sys.modules, self.modules):
            self.adapter.register()
            for dtype in ("nvfp4_v2", "nvfp4_v2_gs"):
                config = self.omni()
                config.data_type = dtype
                self.assertEqual(config.get_quant_method("layer", "prefix"), "inc-method")
                self.base_call.assert_called_with(config, "layer", "prefix")
            for dtype in ("nv_fp", "mx_fp", "int", "future-format"):
                config = self.omni()
                config.data_type = dtype
                self.assertEqual(config.get_quant_method("layer", "prefix"), "omni-method")
                self.omni_call.assert_called_with(config, "layer", "prefix")
        self.assertIs(self.base.get_quant_method, original_base)
        self.assertEqual(self.base.SUPPORTED_DTYPES, {"int", "mx_fp", "nv_fp"})
        self.assertEqual(self.base.SUPPORTED_FORMATS, {"auto_round:llm_compressor"})
        self.assertTrue(self.adapter._E5M3_DTYPES <= self.omni.SUPPORTED_DTYPES)
        self.assertTrue(self.adapter._E5M3_FORMATS <= self.omni.SUPPORTED_FORMATS)
        self.register_inc.assert_called_once_with()

    def test_registration_is_idempotent_even_after_module_reload(self):
        with mock.patch.dict(sys.modules, self.modules):
            self.adapter.register()
            first = self.omni.get_quant_method
            load_adapter().register()
            self.assertIs(self.omni.get_quant_method, first)
            config = self.omni()
            config.data_type = "nv_fp"
            config.get_quant_method(None, "prefix")
            self.omni_call.assert_called_once()

    def test_absent_omni_is_optional_but_broken_dependencies_are_not_hidden(self):
        original_import = __import__
        for missing in ("vllm_omni", "some_omni_dependency"):
            def guarded(name, *args, **kwargs):
                if name == "vllm_omni.quantization.inc_config":
                    raise ModuleNotFoundError(name=missing)
                return original_import(name, *args, **kwargs)

            with self.subTest(missing=missing), mock.patch("builtins.__import__", side_effect=guarded):
                if missing == "vllm_omni":
                    self.adapter.register()
                else:
                    with self.assertRaises(ModuleNotFoundError):
                        self.adapter.register()
        self.register_inc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
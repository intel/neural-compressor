# SPDX-License-Identifier: Apache-2.0
"""CPU-only FLUX.3 validation and existing-family generation regressions."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import benchmark
from benchmark import configure_flux3_package as configure_package
from benchmark import load_prompt_cache


class Flux3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = {"model": "flux-3-image-klein", "dit": {"flow_map": True}, "sampler": {"guidance": None}}
        self.write("config.json", self.package)
        for name in ("model.safetensors", "ae.safetensors", "text_encoder"):
            (self.root / name).touch()

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def test_bf16_native_only(self):
        settings = {}
        self.assertEqual(configure_package(self.root, settings, None), "bf16")
        self.assertEqual(settings["model_class_name"], "Flux3Pipeline")
        self.assertNotIn("quantization_config", settings)

    def test_reject_guidance_and_adapter(self):
        for settings, guidance in (
            ({}, 0.0),
            ({"diffusion_load_format": "diffusers"}, None),
            ({"model_class_name": "DiffusersAdapterPipeline"}, None),
        ):
            with self.subTest(settings=settings, guidance=guidance), self.assertRaises(ValueError):
                configure_package(self.root, settings, guidance)

    def test_reject_wrong_package(self):
        self.package["dit"]["flow_map"] = False
        self.write("config.json", self.package)
        with self.assertRaises(ValueError):
            configure_package(self.root, {}, None)

    def test_missing_asset(self):
        (self.root / "ae.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            configure_package(self.root, {}, None)

    def test_prepared_quantization_config_is_used_unchanged(self):
        quant = {
            "quant_method": "auto-round",
            "bits": 8,
            "group_size": 32,
            "data_type": "mx_fp",
            "packing_format": "auto_round:llm_compressor",
            "block_name_to_quantize": "single_blocks.0.q_proj,early_stream_modulations.image.lin,final_layer.image.linear",
        }
        self.write("quantization_config.json", quant)
        before = (self.root / "quantization_config.json").read_bytes()
        fake = types.SimpleNamespace(safe_open=lambda *a, **k: self.fail("Benchmark must not inspect weights"))
        settings = {}
        with patch.dict(sys.modules, {"safetensors": fake}):
            self.assertEqual(configure_package(self.root, settings, None), "mxfp8")
        self.assertEqual(settings["quantization_config"], quant)
        self.assertEqual((self.root / "quantization_config.json").read_bytes(), before)
        with self.assertRaises(ValueError):
            configure_package(self.root, {"quantization_config": quant}, None)

    def test_reject_other_quantization(self):
        self.write("quantization_config.json", {"quant_method": "fp8"})
        with self.assertRaises(ValueError):
            configure_package(self.root, {}, None)

    def test_cache_integrity(self):
        manifest = b"original manifest"
        manifests = {"drawbench": ([{"id": 0, "caption": "original"}], manifest)}
        cache = {"manifest_sha256": hashlib.sha256(manifest).hexdigest(), "captions": {"0": "expanded"}}
        self.write("cache.json", cache)
        captions, metadata = load_prompt_cache(self.root / "cache.json", manifests)
        self.assertEqual(captions, {"0": "expanded"})
        self.assertEqual(metadata["prompt_policy"], "cached_expansion_score_original")
        for field, value in (("manifest_sha256", "wrong"), ("captions", {}), ("captions", {"0": ""})):
            self.write("cache.json", cache | {field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                load_prompt_cache(self.root / "cache.json", manifests)
        self.assertEqual(load_prompt_cache(None, manifests)[1], {"prompt_policy": "original"})

    def test_all_existing_families_unchanged_and_flux3_omits_guidance(self):
        self.write("model_index.json", {"_class_name": "TestPipeline"})
        (self.root / "prompts.tsv").write_text("id\tcaption\n0\toriginal\n")
        for family, (steps, guidance) in benchmark.PRESETS.items():
            with self.subTest(family=family):
                calls = {}

                class Sampling:
                    def __init__(self, **kwargs):
                        calls["sampling"] = kwargs

                class Omni:
                    default_sampling_params_list = []

                    def __init__(self, **kwargs):
                        calls["engine"] = kwargs

                    def generate(self, prompt, **kwargs):
                        calls["prompt"] = prompt
                        return [types.SimpleNamespace(images=[Image.new("RGB", (16, 16))])]

                    def close(self):
                        calls["closed"] = True

                modules = {
                    "vllm_omni.entrypoints.omni": types.SimpleNamespace(Omni=Omni),
                    "vllm_omni.inputs.data": types.SimpleNamespace(OmniDiffusionSamplingParams=Sampling),
                }
                args = types.SimpleNamespace(
                    output_dir=self.root / family,
                    engine_kwargs="{}",
                    family=family,
                    model_path=str(self.root),
                    steps=None,
                    guidance=None,
                    revision=None,
                    manifest=[("drawbench", str(self.root / "prompts.tsv"))],
                    dtype="bf16",
                    height=16,
                    width=16,
                    seed=42,
                    prompt_cache=None,
                )
                if family == "flux3-klein":
                    cache = {
                        "manifest_sha256": hashlib.sha256((self.root / "prompts.tsv").read_bytes()).hexdigest(),
                        "captions": {"0": "expanded"},
                    }
                    self.write("cache.json", cache)
                    args.prompt_cache = self.root / "cache.json"
                with patch.dict(sys.modules, modules), patch.object(
                    benchmark.importlib.metadata, "version", return_value="test"
                ):
                    benchmark.generate(args)
                self.assertEqual(calls["sampling"]["num_inference_steps"], steps)
                if family == "flux3-klein":
                    self.assertNotIn("guidance_scale", calls["sampling"])
                    self.assertEqual(calls["engine"]["model_class_name"], "Flux3Pipeline")
                else:
                    self.assertEqual(calls["sampling"]["guidance_scale"], guidance)
                    self.assertNotIn("model_class_name", calls["engine"])
                self.assertEqual(calls["prompt"]["prompt"], "expanded" if family == "flux3-klein" else "original")
                self.assertTrue(calls["closed"])
                run = self.root / family / "drawbench"
                self.assertEqual((run / "prompts.tsv").read_bytes(), (self.root / "prompts.tsv").read_bytes())
                self.assertTrue((run / "images/0.png").exists())
                # Resume must not instantiate the engine or regenerate images.
                with patch.dict(sys.modules, modules), patch.object(
                    benchmark.importlib.metadata, "version", return_value="test"
                ):
                    with patch.object(Omni, "__init__", side_effect=AssertionError("Unexpected reload")):
                        benchmark.generate(args)
                    if family == "flux3-klein":
                        cache["captions"]["0"] = "changed expansion"
                        self.write("cache.json", cache)
                        with self.assertRaisesRegex(ValueError, "Run settings changed"):
                            benchmark.generate(args)


if __name__ == "__main__":
    unittest.main()

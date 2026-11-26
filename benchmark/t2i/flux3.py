# SPDX-License-Identifier: Apache-2.0
"""Native FLUX.3 package handling; imported only for the FLUX.3 family."""

import hashlib
import json


def configure_package(path, settings, guidance):
    """Validate a distilled BFL package and configure native Omni loading."""
    package = json.loads((path / "config.json").read_text())
    if package.get("model") != "flux-3-image-klein" or not package.get("dit", {}).get("flow_map"):
        raise ValueError("flux3-klein requires a distilled FLUX.3 Image Klein package")
    if guidance is not None or package.get("sampler", {}).get("guidance") is not None:
        raise ValueError("Distilled FLUX.3 does not support --guidance; omit it")
    if package.get("quantization") is not None:
        raise ValueError("Use a shared FLUX.3 package root with an AutoRound sidecar, not a variant")
    for name in ("model.safetensors", "ae.safetensors", "text_encoder"):
        if not (path / name).exists():
            raise FileNotFoundError(f"Missing native FLUX.3 package asset: {path / name}")
    for key, value in (("model_class_name", "Flux3Pipeline"), ("diffusion_load_format", "default")):
        if settings.setdefault(key, value) != value:
            raise ValueError(f"FLUX.3 requires native {key}={value!r}")
    settings.setdefault("init_timeout", 1200)
    settings.setdefault("stage_init_timeout", 1200)
    sidecar = path / "quantization_config.json"
    if not sidecar.exists():
        return "bf16"
    quant = json.loads(sidecar.read_text())
    expected = {
        "quant_method": "auto-round",
        "bits": 8,
        "group_size": 32,
        "data_type": "mx_fp",
        "packing_format": "auto_round:llm_compressor",
    }
    if not isinstance(quant, dict) or any(quant.get(key) != value for key, value in expected.items()):
        raise ValueError("Native FLUX.3 requires an AutoRound MXFP8 checkpoint")
    if "quantization" in settings or "quantization_config" in settings:
        raise ValueError("FLUX.3 reads quantization from its checkpoint; do not override it")
    from safetensors import safe_open

    # Model-free block metadata can omit quantized modulation/output linears.
    # Read tensor names, not values; never dequantize or requantize the weights.
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as weights:
        modules = sorted(
            name.removesuffix(".weight_scale") for name in weights.keys() if name.endswith(".weight_scale")
        )
    if not modules:
        raise ValueError("MXFP8 checkpoint has no serialized weight scales")
    quant["block_name_to_quantize"] = ",".join(modules)
    settings["quantization_config"] = quant
    return "mxfp8"


def load_prompt_cache(path, manifests):
    """Reuse expanded prompts while keeping original captions for scoring."""
    if path is None:
        return {}, {"prompt_policy": "original"}
    if len(manifests) != 1:
        raise ValueError("--prompt-cache requires exactly one --manifest")
    data = path.expanduser().read_bytes()
    cache = json.loads(data)
    rows, manifest = next(iter(manifests.values()))
    if cache.get("manifest_sha256") != hashlib.sha256(manifest).hexdigest():
        raise ValueError("Prompt cache manifest hash mismatch")
    captions = cache.get("captions", {})
    if not isinstance(captions, dict) or set(captions) != {str(row["id"]) for row in rows}:
        raise ValueError("Prompt cache must contain exactly the manifest IDs")
    if any(not isinstance(text, str) or not text.strip() for text in captions.values()):
        raise ValueError("Cached prompts must be nonempty strings")
    return captions, {
        "prompt_policy": "cached_expansion_score_original",
        "prompt_cache_sha256": hashlib.sha256(data).hexdigest(),
    }

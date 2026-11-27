#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare prompt manifests and generate matching PNGs using vLLM-Omni."""

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import random
import re
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

DRAWBENCH = "sayakpaul/drawbench"
DRAWBENCH_REVISION = "ad794cba698c253ffdb6d18eff3fc87b004c1135"
COCO_CAPTIONS = "annotations/captions_val2014.json"
FIELDS = ("id", "caption", "category", "source_id", "caption_id")
PRESETS = {
    "flux1-dev": (20, 3.5),
    "flux1-schnell": (4, 0.0),
    "flux2-dev": (50, 4.0),
    "flux2-klein": (4, 1.0),
    "z-image-turbo": (8, 0.0),
    "flux3-klein": (16, None),
}
FAMILY_PATTERNS = {
    "flux1-dev": "flux.1-dev",
    "flux1-schnell": "flux.1-schnell",
    "flux2-dev": "flux.2-dev",
    "flux2-klein": "flux.2-klein",
    "z-image-turbo": "z-image-turbo",
    "flux3-klein": "flux-3-image-klein",
}
FLUX2_QUANTIZED_COMPONENTS = ("transformer", "text_encoder")
PRECISIONS = ("bf16", "mxfp8")


def configure_flux3_package(path, settings, guidance):
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
    # The native package and complete quantization scope must be prepared before benchmarking.
    # Pass checkpoint metadata through unchanged; do not inspect or rewrite weights.
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


def atomic_write(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_write(path, (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def csv_bytes(rows, fields, delimiter=","):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def load_manifest(path):
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), delimiter="\t")
    if not {"id", "caption"}.issubset(reader.fieldnames or []):
        raise ValueError(f"{path} requires TSV columns id and caption")
    rows = []
    seen = set()
    for row in reader:
        row_id = int(row["id"])
        caption = row["caption"]
        if row_id < 0 or row_id in seen or not caption or not caption.strip():
            raise ValueError(f"Invalid or duplicate prompt id {row_id} in {path}")
        seen.add(row_id)
        rows.append({key: row.get(key) or "" for key in FIELDS} | {"id": row_id})
    if not rows:
        raise ValueError(f"No prompts in {path}")
    return rows, data


def drawbench_rows(source):
    if source is None:
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                DRAWBENCH, "DrawBench Prompts - Sheet1.csv", repo_type="dataset", revision=DRAWBENCH_REVISION
            )
        )
    with source.open(encoding="utf-8-sig", newline="") as file:
        return [
            {"id": i, "caption": row["Prompts"], "category": row["Category"], "source_id": i}
            for i, row in enumerate(csv.DictReader(file))
        ]


def coco_rows(source, count, seed):
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive, archive.open(COCO_CAPTIONS) as file:
            annotations = json.load(file)["annotations"]
    else:
        with source.open(encoding="utf-8") as file:
            annotations = json.load(file)["annotations"]
    by_image = {}
    for item in annotations:
        image_id = item["image_id"]
        if image_id not in by_image or item["id"] < by_image[image_id]["id"]:
            by_image[image_id] = item
    if count > len(by_image):
        raise ValueError(f"Requested {count} image IDs from {len(by_image)} annotated COCO images")
    ids = random.Random(seed).sample(sorted(by_image), count)
    return [
        {
            "id": i,
            "caption": by_image[image_id]["caption"].strip(),
            "source_id": image_id,
            "caption_id": by_image[image_id]["id"],
        }
        for i, image_id in enumerate(ids)
    ]


def prepare(args):
    if args.dataset == "drawbench":
        rows = drawbench_rows(args.source)
        if args.count:
            if args.count > len(rows):
                raise ValueError(f"Requested {args.count} prompts from {len(rows)} DrawBench rows")
            rows = rows[: args.count]
    else:
        if args.source is None:
            raise ValueError("COCO2014 requires --source captions_val2014.json or annotations_trainval2014.zip")
        rows = coco_rows(args.source, args.count or 5000, args.dataset_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, csv_bytes(rows, FIELDS, delimiter="\t"))
    print(f"Wrote {len(rows)} {args.dataset} prompts to {args.output}")


def checkpoint(args):
    local = Path(args.model_path).expanduser()
    if local.is_dir():
        if args.revision:
            raise ValueError("--revision applies to Hugging Face repository IDs")
        return str(local.resolve()), None

    from huggingface_hub import HfApi, get_token, snapshot_download

    token = get_token()
    revision = HfApi().model_info(args.model_path, revision=args.revision, token=token).sha
    # Pin the actual files to the commit recorded in the run configuration.
    return snapshot_download(args.model_path, revision=revision, token=token), revision


def checkpoint_precision(model_path):
    index = json.loads((model_path / "model_index.json").read_text())
    quantized = False
    for component, entry in index.items():
        if component.startswith("_") or not isinstance(entry, list) or not entry or not entry[0]:
            continue
        directory = model_path / component
        config_path = directory / "config.json"
        sidecar = directory / "quantization_config.json"
        inline = json.loads(config_path.read_text()).get("quantization_config") if config_path.is_file() else None
        for quant in (json.loads(sidecar.read_text()) if sidecar.is_file() else None, inline):
            if quant is None:
                continue
            if not isinstance(quant, dict) or (
                quant.get("quant_method"),
                quant.get("data_type"),
                quant.get("bits"),
            ) != ("auto-round", "mx_fp", 8):
                raise ValueError(f"{component} uses an unsupported quantization format")
            quantized = True
    return "mxfp8" if quantized else "bf16"


def flux2_component_configs(model_path):
    configs = {}
    for component in FLUX2_QUANTIZED_COMPONENTS:
        config_path = model_path / component / "quantization_config.json"
        if not config_path.is_file():
            return None
        config = json.loads(config_path.read_text())
        if (config.get("quant_method"), config.get("data_type"), config.get("bits")) != ("auto-round", "mx_fp", 8):
            return None
        configs[component] = config
    return configs


def flux2_component_quantization(model_path, configs):
    from safetensors import safe_open

    for component, config in configs.items():
        directory = model_path / component
        matrix_weights = set()
        scaled_weights = set()
        for shard in directory.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.endswith(".weight_scale"):
                        scaled_weights.add(key.removesuffix(".weight_scale"))
                    elif key.endswith(".weight") and len(handle.get_slice(key).get_shape()) == 2:
                        matrix_weights.add(key.removesuffix(".weight"))
        # These exports mix MXFP8 and BF16 modules. Preserve unscaled matrices as BF16.
        config["block_name_to_quantize"] = None
        config["extra_config"] = {
            f"^{re.escape(component + '.' + layer)}$": {
                "bits": 16,
                "data_type": "float",
                "act_bits": 16,
                "act_data_type": "float",
            }
            for layer in sorted(matrix_weights - scaled_weights)
        }
    return configs | {"vae": None}


def flux2_checkpoint_view(model_path, stack):
    view = Path(stack.enter_context(TemporaryDirectory(prefix="flux2-eval-")))
    for child in model_path.iterdir():
        target = view / child.name
        if child.name in FLUX2_QUANTIZED_COMPONENTS:
            target.mkdir()
            has_model_index = (child / "model.safetensors.index.json").is_file()
            for file in child.iterdir():
                if not has_model_index or file.name != "diffusion_pytorch_model.safetensors.index.json":
                    (target / file.name).symlink_to(file, target_is_directory=file.is_dir())
        else:
            target.symlink_to(child, target_is_directory=child.is_dir())
    return str(view)


def open_run_lock(run, stack):
    run.mkdir(parents=True, exist_ok=True)
    lock = stack.enter_context((run / ".lock").open("a"))
    fcntl.flock(lock, fcntl.LOCK_EX)


def generate(args):
    from PIL import Image

    output_dir = args.output_dir.expanduser()
    settings = json.loads(args.engine_kwargs)
    if not isinstance(settings, dict) or "model" in settings:
        raise ValueError("--engine-kwargs must be a JSON object without a model field")

    family = args.family or next(
        (name for name, pattern in FAMILY_PATTERNS.items() if pattern in args.model_path.lower()), None
    )
    if family is None and (args.steps is None or args.guidance is None):
        raise ValueError("Specify --family or both --steps and --guidance for this model")
    default_steps, default_guidance = PRESETS.get(family, (None, None))
    prompt_cache = getattr(args, "prompt_cache", None)
    if prompt_cache is not None and family != "flux3-klein":
        raise ValueError("--prompt-cache is currently supported only for --family flux3-klein")
    if family == "flux2-dev":
        # Large text encoder and transformer shards can exceed the default on shared storage.
        settings.setdefault("init_timeout", 2400)
        settings.setdefault("stage_init_timeout", 2400)
    runtime = {name: importlib.metadata.version(name) for name in ("vllm", "vllm-omni", "torch")}
    manifests = {}
    for dataset, manifest_path in args.manifest:
        if dataset not in ("drawbench", "coco2014") or dataset in manifests:
            raise ValueError(f"Invalid or duplicate dataset label {dataset!r}")
        rows, data = load_manifest(Path(manifest_path).expanduser())
        manifests[dataset] = (rows, data)

    print(f"Resolving checkpoint {args.model_path}", flush=True)
    model_path, revision = checkpoint(args)
    checkpoint_path = Path(model_path).resolve()
    prompt_overrides, prompt_metadata = {}, {}
    if family == "flux3-klein":
        precision = configure_flux3_package(checkpoint_path, settings, args.guidance)
        prompt_overrides, prompt_metadata = load_prompt_cache(prompt_cache, manifests)
    else:
        precision = checkpoint_precision(checkpoint_path)
    if args.dtype is not None and args.dtype != precision:
        raise ValueError(f"--dtype={args.dtype} does not match the checkpoint precision ({precision})")
    if precision == "bf16":
        dtype = settings.setdefault("dtype", "bfloat16")
        quantized = any(settings.get(key) is not None for key in ("quantization", "quantization_config"))
        if dtype not in ("bf16", "bfloat16") or quantized:
            raise ValueError("BF16 requires BF16 inference and no quantization config")
    else:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    component_configs = flux2_component_configs(checkpoint_path) if family == "flux2-dev" else None
    if component_configs and "quantization_config" not in settings:
        print("Configuring FLUX.2-dev quantization for the transformer and text encoder", flush=True)
        settings["quantization_config"] = flux2_component_quantization(checkpoint_path, component_configs)
    config = {
        "model": args.model_path if revision else model_path,
        "revision": revision,
        "dtype": precision,
        "backend": "vllm-omni",
        "runtime": runtime,
        "engine_kwargs": settings,
        "cuda_launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING"),
        "steps": args.steps or default_steps,
        "guidance": args.guidance if args.guidance is not None else default_guidance,
        "height": args.height,
        "width": args.width,
        "base_seed": args.seed,
        **prompt_metadata,
    }
    with ExitStack() as lock:
        open_run_lock(output_dir, lock)
        model_config = output_dir / "model.json"
        if model_config.exists() and json.loads(model_config.read_text()) != config:
            raise ValueError(f"Run settings changed for {output_dir}. Choose a new --output_dir.")
        atomic_json(model_config, config)

    with ExitStack() as engine:
        omni = None
        for dataset, (rows, data) in manifests.items():
            run = output_dir / dataset
            images = run / "images"
            run_config = config | {"dataset": dataset, "manifest_sha256": hashlib.sha256(data).hexdigest()}
            with ExitStack() as lock:
                open_run_lock(run, lock)
                images.mkdir(exist_ok=True)
                config_path = run / "run.json"
                if config_path.exists():
                    if json.loads(config_path.read_text()) != run_config:
                        raise ValueError(f"Run settings changed for {run}. Choose a new --output_dir.")
                elif any(images.glob("*.png")):
                    raise ValueError(f"No run.json for existing images in {run}")
                atomic_write(run / "prompts.tsv", data)
                atomic_json(config_path, run_config)

                for row in rows:
                    row_id = row["id"]
                    path = images / f"{row_id}.png"
                    if path.exists():
                        try:
                            with Image.open(path) as image:
                                if image.size == (config["width"], config["height"]):
                                    image.verify()
                                    continue
                        except (OSError, SyntaxError):
                            pass
                    if omni is None:
                        print(f"Loading vLLM-Omni for {args.model_path} ({precision})", flush=True)
                        from vllm_omni.entrypoints.omni import Omni
                        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

                        omni_model = flux2_checkpoint_view(checkpoint_path, engine) if component_configs else model_path
                        omni = Omni(model=omni_model, **settings)
                        engine.callback(omni.close)
                    seed = args.seed + row_id
                    guidance_kwargs = {} if family == "flux3-klein" else {"guidance_scale": config["guidance"]}
                    params = OmniDiffusionSamplingParams(
                        height=config["height"],
                        width=config["width"],
                        seed=seed,
                        **guidance_kwargs,
                        num_inference_steps=config["steps"],
                        num_outputs_per_prompt=1,
                    )
                    defaults = omni.default_sampling_params_list
                    sampling = (
                        [
                            params if isinstance(default, OmniDiffusionSamplingParams) else default
                            for default in defaults
                        ]
                        if len(defaults or []) > 1
                        else [params]
                    )
                    started = time.perf_counter()
                    outputs = omni.generate(
                        {"prompt": prompt_overrides.get(str(row_id), row["caption"]), "modalities": ["image"]},
                        sampling_params_list=sampling,
                        use_tqdm=False,
                    )
                    generated = [image for output in outputs for image in (output.images or [])]
                    if len(generated) != 1 or generated[0].size != (config["width"], config["height"]):
                        raise RuntimeError(
                            f"Expected one {config['width']}x{config['height']} image for {dataset}/{row_id}"
                        )
                    png = io.BytesIO()
                    generated[0].save(png, format="PNG")
                    atomic_write(path, png.getvalue())
                    print(f"{dataset}/{row_id} seed={seed} {time.perf_counter() - started:.2f}s -> {path}", flush=True)
                print(f"{run}: {len(rows)} images ready", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="prepare a reusable DrawBench or COCO2014 TSV")
    prep.add_argument("dataset", choices=("drawbench", "coco2014"))
    prep.add_argument("--source", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--count", type=positive_int)
    prep.add_argument("--dataset-seed", type=int, default=42)
    prep.set_defaults(func=prepare)

    gen = commands.add_parser("generate", help="generate images for a checkpoint with vLLM-Omni")
    gen.add_argument("--model_path", required=True, help="local checkpoint or Hugging Face model ID")
    gen.add_argument("--revision")
    gen.add_argument("--manifest", nargs=2, metavar=("DATASET", "TSV"), action="append", required=True)
    gen.add_argument("--output_dir", type=Path, required=True)
    gen.add_argument("--dtype", choices=PRECISIONS, help="check the expected checkpoint precision")
    gen.add_argument("--family", choices=PRESETS)
    gen.add_argument("--steps", type=positive_int)
    gen.add_argument("--guidance", type=float)
    gen.add_argument("--height", type=positive_int, default=1024)
    gen.add_argument("--width", type=positive_int, default=1024)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--engine-kwargs", default="{}")
    gen.add_argument("--prompt-cache", type=Path, help="FLUX.3 expanded-caption JSON; score original manifest captions")
    gen.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

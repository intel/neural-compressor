#!/usr/bin/env python3
"""Fixed-input Wan2.2 benchmarks. Prepare/verify/help require only Python stdlib.

Public evaluator helpers: SETTINGS, METRICS, build_jobs(task, images=None),
atomic_json(path, value), sha256(path), fingerprint(configuration, job=None),
probe(path, frames, width, height, fps=16), valid(path, width, height), and
verify(root, task). Verification raises on incomplete or inconsistent runs.
Weight bytes are NOT fingerprinted: local checkpoint weights must be immutable.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
SOURCE_COMMIT = "7465c0f52bcb46a55f62c1b8613846cda30d0083"
I2V_EXAMPLE = "examples/offline_inference/image_to_video/image_to_video.py"
EXPECTED_COUNTS = {"t2v": 165, "i2v": 355}
DATA_FILES = {"t2v": "t2v_prompts.json", "i2v": "i2v_metadata.json"}
GROUPS = ("subject_consistency", "overall_consistency")
METRICS = {
    "t2v": ("imaging_quality", "aesthetic_quality", "motion_smoothness", "overall_consistency", "subject_consistency"),
    "i2v": ("i2v_subject", "subject_consistency", "background_consistency", "motion_smoothness", "i2v_background"),
}
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
NEGATIVE = NEGATIVE_PROMPT
_COMMON = dict(
    frames=81,
    fps=16,
    steps=40,
    negative_prompt=NEGATIVE_PROMPT,
    attention="TRTLLM_ATTN",
    enforce_eager=True,
    cache_backend=None,
    enable_cpu_offload=False,
    enable_layerwise_offload=False,
    vae_use_tiling=False,
    vae_use_slicing=False,
    samples_per_prompt=1,
)
SETTINGS = {
    "t2v": dict(
        _COMMON,
        width=1280,
        height=720,
        seed=42,
        guidance_scale=4.0,
        guidance_scale_2=3.0,
        boundary_ratio=0.875,
        flow_shift=5.0,
        linear_backend="flashinfer_cutlass",
        init_timeout=3600,
        stage_init_timeout=3600,
        enable_diffusion_pipeline_profiler=True,
    ),
    "i2v": dict(
        _COMMON,
        width=832,
        height=480,
        seed=0,
        guidance_scale=3.5,
        guidance_scale_2=3.5,
        boundary_ratio=0.9,
        flow_shift=3.0,
        dimension_multiple=16,
        image_resize="LANCZOS (official Omni example)",
        init_timeout=7200,
        stage_init_timeout=7200,
        enable_diffusion_pipeline_profiler=False,
    ),
}
MODEL_CLASSES = {"t2v": "WanPipeline", "i2v": "WanImageToVideoPipeline"}
PACKAGES = (
    "torch",
    "numpy",
    "vllm",
    "vllm-omni",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "diffusers",
    "transformers",
    "vbench",
)


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(configuration, job=None):
    """Canonical JSON SHA-256; a job fingerprint binds both config and input."""
    value = configuration if job is None else dict(configuration=configuration, job=job)
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    """Durably replace JSON using a unique temporary file on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            tmp = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _immutable(path, value):
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"Refusing mismatched resume: {path}")
    else:
        atomic_json(path, value)


def _task_name(task):
    if task not in SETTINGS:
        raise ValueError(f"Unknown task: {task!r}")
    return task


def build_jobs(task, images=None):
    """Preserve snapshot order and duplicates; select the I2V 355-record union."""
    _task_name(task)
    records = load_json(DATA / DATA_FILES[task])
    if not isinstance(records, list):
        raise ValueError("Input snapshot must be a JSON list")
    if task == "i2v":
        if len(records) != 1118:
            raise ValueError(f"Expected 1118 original I2V records, got {len(records)}")
        if images is None:
            raise ValueError("I2V requires --i2v-images pointing to crop/16-9")
        images = Path(images).expanduser().resolve(strict=True)
        if not images.is_dir():
            raise ValueError(f"Not an image directory: {images}")
        for record in records:
            dimensions = record.get("dimension")
            if not isinstance(dimensions, list) or not all(isinstance(d, str) for d in dimensions):
                raise ValueError("I2V dimensions must be a list of strings")
        records = [r for r in records if {"i2v_subject", "i2v_background"}.intersection(r["dimension"])]
    if len(records) != EXPECTED_COUNTS[task]:
        raise ValueError(f"Expected {EXPECTED_COUNTS[task]} {task} jobs, got {len(records)}")
    settings = SETTINGS[task]
    jobs = []
    for index, record in enumerate(records):
        prompt = record.get("prompt" if task == "t2v" else "prompt_en")
        if not isinstance(prompt, str) or not prompt.strip() or "\0" in prompt:
            raise ValueError(f"Invalid prompt at index {index}")
        job = dict(
            index=index, prompt=prompt, filename=f"{index:04d}.mp4", width=settings["width"], height=settings["height"]
        )
        if task == "t2v":
            if record.get("group") not in GROUPS:
                raise ValueError(f"Invalid T2V group at index {index}")
            job["group"] = record["group"]
        else:
            name = record.get("image_name")
            if not isinstance(name, str) or not name or Path(name).name != name or "\0" in name:
                raise ValueError(f"Unsafe image_name at index {index}")
            image = (images / name).resolve(strict=True)
            if not image.is_relative_to(images) or not image.is_file() or image.stat().st_size == 0:
                raise ValueError(f"Missing, empty, or escaping reference image: {image}")
            job.update(dimensions=record["dimension"], image_name=name, image=str(image), image_sha256=sha256(image))
        jobs.append(job)
    if task == "t2v":
        counts = {g: sum(j["group"] == g for j in jobs) for g in GROUPS}
        if counts != dict(subject_consistency=72, overall_consistency=93):
            raise ValueError(f"Unexpected T2V source-group counts: {counts}")
    else:
        counts = {m: sum(m in j["dimensions"] for j in jobs) for m in METRICS[task]}
        expected = dict(
            i2v_subject=246,
            subject_consistency=246,
            background_consistency=109,
            motion_smoothness=246,
            i2v_background=109,
        )
        if counts != expected:
            raise ValueError(f"Unexpected official I2V dimension counts: {counts}")
    return jobs


def _task_record(task, jobs, images):
    return dict(
        schema_version=1,
        task=task,
        protocol="selected-inputs-one-fixed-seed-sample-not-full-standard-vbench",
        selected_count=EXPECTED_COUNTS[task],
        samples_per_input=1,
        data_file="data/" + DATA_FILES[task],
        data_sha256=sha256(DATA / DATA_FILES[task]),
        jobs_sha256=fingerprint(jobs),
        i2v_images=str(Path(images).expanduser().resolve()) if images is not None else None,
        settings=SETTINGS[task],
    )


def prepare(root, task, images=None):
    root = Path(root).expanduser().resolve()
    if task == "t2v" and images is not None:
        raise ValueError("--i2v-images is only valid for I2V")
    jobs = build_jobs(task, images)
    task_record = _task_record(task, jobs, images)
    for folder in ("logs", "metadata", "locks", "videos"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    with _lock(root / "locks" / "state.lock"):
        # Check both before writing either: mismatches never change a snapshot.
        for name, value in (("task.json", task_record), ("jobs.json", jobs)):
            if (root / name).exists() and load_json(root / name) != value:
                raise RuntimeError(f"Refusing mismatched preparation: {root / name}")
        _immutable(root / "jobs.json", jobs)
        _immutable(root / "task.json", task_record)
    print(f"PREPARED {task} {len(jobs)} inputs: {root}", flush=True)
    return jobs


def read_prepared(root, task):
    """Validate persisted jobs against current repository data and image hashes."""
    _task_name(task)
    root = Path(root).expanduser().resolve()
    task_record = load_json(root / "task.json")
    if task_record.get("task") != task:
        raise RuntimeError("Run root belongs to a different task")
    jobs = build_jobs(task, task_record.get("i2v_images"))
    if task_record != _task_record(task, jobs, task_record.get("i2v_images")):
        raise RuntimeError("Task protocol/data/image snapshot changed; use a new root")
    if load_json(root / "jobs.json") != jobs:
        raise RuntimeError("Immutable jobs.json differs from current data/reference images")
    return task_record, jobs


def probe(path, frames=81, width=None, height=None, fps=16):
    """Return True only for a decodable video with exactly the requested format."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        check = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        stream = json.loads(check.stdout)["streams"][0]
        return (
            check.returncode == 0
            and not check.stderr.strip()
            and int(stream["width"]) == width
            and int(stream["height"]) == height
            and int(stream["nb_read_frames"]) == frames
            and Fraction(stream["r_frame_rate"]) == Fraction(str(fps))
            and Fraction(stream["avg_frame_rate"]) == Fraction(str(fps))
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required on PATH") from exc
    except (subprocess.TimeoutExpired, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OSError):
        return False


def valid(path, width, height, frames=81, fps=16):
    return probe(path, frames, width, height, fps)


def _git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True, stderr=subprocess.PIPE).strip()


def validate_source(source):
    source = Path(source).expanduser().resolve(strict=True)
    if Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise RuntimeError("--omni-source must be the checkout root")
    commit = _git(source, "rev-parse", "HEAD")
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected Omni commit {SOURCE_COMMIT}, got {commit}")
    scopes = (
        "vllm_omni",
        "examples/offline_inference",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements",
        "requirements.txt",
    )
    dirty = _git(source, "status", "--porcelain", "--untracked-files=no", "--", *scopes)
    if dirty:
        raise RuntimeError(f"Tracked Omni source changes in inference scope:\n{dirty}")
    example = source / I2V_EXAMPLE
    if not example.is_file() or not (source / "vllm_omni" / "__init__.py").is_file():
        raise RuntimeError("Omni source/example missing")
    return dict(
        path=str(source),
        commit=commit,
        clean_tracked_scopes=list(scopes),
        i2v_example=I2V_EXAMPLE,
        i2v_example_sha256=sha256(example),
    )


def _import_omni(source):
    """Prepend the requested checkout BEFORE any Omni import, then check origin."""
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    for name, module in tuple(sys.modules.items()):
        if name == "vllm_omni" or name.startswith("vllm_omni."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(source / "vllm_omni"):
                raise RuntimeError(f"Already imported Omni outside selected checkout: {origin}")
    module = importlib.import_module("vllm_omni")
    if not module.__file__ or not Path(module.__file__).resolve().is_relative_to(source / "vllm_omni"):
        raise RuntimeError(f"Imported Omni outside selected checkout: {module.__file__}")
    return module


def validate_model(model, task, precision):
    """Validate both transformer components and record lightweight weight provenance."""
    model = Path(model).expanduser().resolve(strict=True)
    if not model.is_dir():
        raise ValueError("--model must be a local checkpoint directory, not a Hub ID")
    if precision not in ("bf16", "nvfp4", "mxfp8", "nvfp4_e5m3"):
        raise ValueError(f"Unknown precision: {precision}")
    pipeline = load_json(model / "model_index.json")
    if pipeline.get("_class_name") != MODEL_CLASSES[task]:
        raise RuntimeError(f"Expected {MODEL_CLASSES[task]}, got {pipeline.get('_class_name')}")
    for component in ("transformer", "transformer_2"):
        directory = model / component
        config = load_json(directory / "config.json")
        if config.get("_class_name") != "WanTransformer3DModel":
            raise RuntimeError(f"Unexpected transformer class in {component}")
        quant_file = directory / "quantization_config.json"
        inline = config.get("quantization_config")
        if precision == "bf16":
            if quant_file.exists() or inline is not None:
                raise RuntimeError(f"BF16 requires no quantization config in {component}")
            dtype = config.get("torch_dtype", config.get("dtype"))
            if dtype is not None and dtype not in ("bfloat16", "torch.bfloat16"):
                raise RuntimeError(f"BF16 component declares unexpected dtype: {component}: {dtype}")
        else:
            quant = load_json(quant_file)
            bits, group, data_type, act_type = (
                (4, 16, "nv_fp", "nv_fp4_with_static_gs")
                if precision == "nvfp4"
                else (4, 16, "nvfp4_v2", "nvfp4_v2") if precision == "nvfp4_e5m3" else (8, 32, "mx_fp", "mx_fp")
            )
            expected = dict(
                quant_method="auto-round",
                bits=bits,
                act_bits=bits,
                group_size=group,
                act_group_size=group,
                data_type=data_type,
                act_data_type=act_type,
                sym=True,
                act_sym=True,
            )
            if any(quant.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Expected AutoRound {precision} W{bits}A{bits} group {group}: {component}")
            if precision == "nvfp4_e5m3" and (
                quant.get("packing_format") != "auto_round:llm_compressor_nvfp4_e5m3"
                or quant.get("act_dynamic") is not True
            ):
                raise RuntimeError(f"Expected packed E5M3 with dynamic activations: {component}")
            if inline is not None and inline != quant:
                raise RuntimeError(f"Conflicting inline quantization config: {component}")
        if not any(directory.glob("*.safetensors")):
            raise RuntimeError(f"Missing safetensors weights: {component}")
    weight_roots = [model]
    # Hub snapshots normally symlink their weights to blobs in the same cache repository.
    if model.parent.name == "snapshots" and model.parent.parent.name.startswith("models--"):
        weight_roots.append(model.parent.parent / "blobs")
    json_files = {}
    indexes = {}
    for path in sorted(model.rglob("*.json")):
        relative = path.relative_to(model).as_posix()
        json_files[relative] = dict(sha256=sha256(path), size_bytes=path.stat().st_size)
        if path.name.endswith(".safetensors.index.json"):
            index = load_json(path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise RuntimeError(f"Invalid safetensors weight index: {path}")
            shards = sorted(set(weight_map.values()))
            for name in shards:
                relative_shard = Path(name)
                if relative_shard.is_absolute() or ".." in relative_shard.parts:
                    raise RuntimeError(f"Unsafe indexed weight path: {name}")
                shard = (path.parent / relative_shard).resolve()
                if not shard.is_file() or not any(shard.is_relative_to(root) for root in weight_roots):
                    raise RuntimeError(f"Unsafe/missing indexed weight: {name}")
            indexes[relative] = dict(sha256=json_files[relative]["sha256"], shards=shards)
    weights = {
        p.relative_to(model).as_posix(): p.stat().st_size
        for p in sorted(model.rglob("*"))
        if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt", ".pth")
    }
    if any(size <= 0 for size in weights.values()):
        raise RuntimeError("Checkpoint contains empty weight files")
    return dict(
        path=str(model),
        precision=precision,
        pipeline_class=MODEL_CLASSES[task],
        json_files=json_files,
        model_configuration_sha256=fingerprint(json_files),
        safetensors_indexes=indexes,
        weight_file_sizes=weights,
        weight_provenance="NOT_CONTENT_FINGERPRINTED: sizes and index/config hashes only; "
        "checkpoint weight bytes MUST remain immutable",
    )


def configuration(root, task, model, precision, source):
    task_record, _ = read_prepared(root, task)
    return dict(
        schema_version=1,
        task=task,
        task_sha256=fingerprint(task_record),
        source=validate_source(source),
        model=validate_model(model, task, precision),
        settings=SETTINGS[task],
        sampling_sha256=fingerprint(SETTINGS[task]),
        benchmark_sha256=sha256(Path(__file__)),
    )


def _check_configuration(config, task, task_record):
    if (
        config.get("schema_version") != 1
        or config.get("task") != task
        or config.get("task_sha256") != fingerprint(task_record)
        or config.get("settings") != SETTINGS[task]
        or config.get("sampling_sha256") != fingerprint(SETTINGS[task])
        or config.get("source", {}).get("commit") != SOURCE_COMMIT
        or config.get("model", {}).get("pipeline_class") != MODEL_CLASSES[task]
        or config.get("model", {}).get("precision") not in ("bf16", "nvfp4", "mxfp8", "nvfp4_e5m3")
        or config.get("benchmark_sha256") != sha256(Path(__file__))
    ):
        raise RuntimeError("Persisted generation configuration does not match this protocol/script")


def _job_paths(root, job):
    return root / "videos" / job["filename"], root / "metadata" / f"{job['index']:04d}.json"


def _completed_job(root, job, config):
    video, metadata = _job_paths(root, job)
    if not video.is_file() or not metadata.is_file():
        return False
    try:
        record = load_json(metadata)
        if (
            record.get("fingerprint") != fingerprint(config, job)
            or record.get("configuration_sha256") != fingerprint(config)
            or record.get("job") != job
            or record.get("size_bytes") != video.stat().st_size
            or record.get("video_sha256") != sha256(video)
        ):
            return False
        settings = config["settings"]
        return probe(video, settings["frames"], settings["width"], settings["height"], settings["fps"])
    except (ValueError, KeyError, TypeError, OSError):
        return False


def normalize_output(result):
    """Historical user-authored T2V normalization, with imports kept lazy."""
    import numpy as np
    import torch
    from vllm_omni.outputs import OmniRequestOutput

    if isinstance(result, list):
        result = result[0]
    if isinstance(result, OmniRequestOutput):
        if not result.images:
            raise RuntimeError("No generated video frames")
        result = result.images
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    if isinstance(result, dict):
        result = result.get("frames", result.get("video"))
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu()
        if result.ndim == 5:
            result = result[0]
        if result.ndim == 4 and result.shape[0] in (3, 4):
            result = result.permute(1, 2, 3, 0)
        result = result.float().numpy()
    elif isinstance(result, list):
        result = np.stack(
            [x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x) for x in result]
        )
    result = np.asarray(result)
    if result.ndim == 5:
        result = result[0]
    if result.ndim != 4 or result.shape[-1] != 3:
        raise RuntimeError(f"Unexpected video shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite generated frames")
    if result.dtype == np.uint8:
        result = result.astype(np.float32) / 255
    if result.min() < -0.01 or result.max() > 1.01:
        raise RuntimeError(f"Unexpected decoded range: {result.min()}, {result.max()}")
    return list(np.clip(result, 0, 1))


class _Generator:
    """One lazy engine per worker.

    The official I2V main function is unchanged.
    """

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.engine_kwargs = None
        self.example = None
        self.original_omni = None
        self.gpu = None

    def _initialize_imports(self):
        source = self.config["source"]["path"]
        if validate_source(source) != self.config["source"]:
            raise RuntimeError("Omni source changed after configuration binding")
        _import_omni(source)
        import torch

        self.gpu = dict(
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            logical_device=torch.cuda.current_device(),
            name=torch.cuda.get_device_name(torch.cuda.current_device()),
        )
        if self.config["task"] == "i2v":
            path = Path(source) / I2V_EXAMPLE
            spec = importlib.util.spec_from_file_location("_wan_benchmark_official_i2v", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot import the official I2V example")
            example = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example)
            self.example = example
            self.original_omni = example.Omni
            example.Omni = self._reuse_engine

    def _reuse_engine(self, **kwargs):
        if self.engine is None:
            self.engine = self.original_omni(**kwargs)
            self.engine_kwargs = kwargs.copy()
        elif kwargs != self.engine_kwargs:
            raise RuntimeError("Official example engine arguments changed between inputs")
        return self.engine

    def generate(self, job, output):
        if self.gpu is None:
            self._initialize_imports()
        settings = self.config["settings"]
        model = self.config["model"]["path"]
        if self.config["task"] == "i2v":
            argv = [
                str(Path(self.config["source"]["path"]) / I2V_EXAMPLE),
                "--model",
                model,
                "--model-class-name",
                MODEL_CLASSES["i2v"],
                "--image",
                job["image"],
                "--prompt",
                job["prompt"],
                "--negative-prompt",
                settings["negative_prompt"],
                "--width",
                "832",
                "--height",
                "480",
                "--num-frames",
                "81",
                "--fps",
                "16",
                "--num-inference-steps",
                "40",
                "--guidance-scale",
                "3.5",
                "--guidance-scale-high",
                "3.5",
                "--boundary-ratio",
                "0.9",
                "--flow-shift",
                "3.0",
                "--seed",
                "0",
                "--enforce-eager",
                "--init-timeout",
                "7200",
                "--stage-init-timeout",
                "7200",
                "--output",
                str(output),
            ]
            previous = sys.argv
            try:
                sys.argv = argv
                self.example.main()
            finally:
                sys.argv = previous
            return argv
        import torch
        from diffusers.utils import export_to_video
        from vllm_omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if self.engine is None:
            self.engine = Omni(
                model=model,
                model_class_name="WanPipeline",
                enforce_eager=True,
                init_timeout=3600,
                stage_init_timeout=3600,
                linear_backend="flashinfer_cutlass",
                boundary_ratio=0.875,
                flow_shift=5.0,
                enable_cpu_offload=False,
                enable_layerwise_offload=False,
                vae_use_tiling=False,
                vae_use_slicing=False,
                enable_diffusion_pipeline_profiler=True,
            )
        params = OmniDiffusionSamplingParams(
            height=720,
            width=1280,
            num_frames=81,
            num_inference_steps=40,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            fps=16,
        )
        result = self.engine.generate({"prompt": job["prompt"], "negative_prompt": settings["negative_prompt"]}, params)
        export_to_video(normalize_output(result), str(output), fps=16)
        return None

    def close(self):
        try:
            if self.engine is not None:
                self.engine.close()
        finally:
            if self.example is not None:
                self.example.Omni = self.original_omni


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def generate(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        raise ValueError("Require num-shards >= 1, 0 <= shard-index < num-shards, limit >= 0")
    root = Path(args.root).expanduser().resolve()
    task_record, jobs = read_prepared(root, args.task)
    cfg = configuration(root, args.task, args.model, args.precision, args.omni_source)
    with _lock(root / "locks" / "state.lock"):
        if not (root / "configuration.json").exists():
            if any((root / "metadata").glob("*.json")) or any((root / "videos").glob("*.mp4")):
                raise RuntimeError("Existing outputs without configuration; refusing to rebind")
        _immutable(root / "configuration.json", cfg)
    _check_configuration(cfg, args.task, task_record)
    # This must precede all runtime imports, including the official example import.
    old_attention = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TRTLLM_ATTN"
    runner = _Generator(cfg)
    count = 0
    skipped = 0
    log_path = root / "logs" / f"worker-{args.shard_index:03d}-{os.getpid()}.jsonl"

    def event(kind, **fields):
        record = dict(event=kind, time=_now(), shard_index=args.shard_index, num_shards=args.num_shards, **fields)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        print(kind, json.dumps(fields, ensure_ascii=False), flush=True)

    try:
        event(
            "WORKER_START",
            configuration_sha256=fingerprint(cfg),
            limit=args.limit,
            expected_total=len(jobs),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        for job in jobs:
            if job["index"] % args.num_shards != args.shard_index:
                continue
            index = job["index"]
            with _lock(root / "locks" / f"{index:04d}.lock"):
                if args.task == "i2v" and sha256(job["image"]) != job["image_sha256"]:
                    raise RuntimeError(f"Reference image changed: {job['image_name']}")
                if _completed_job(root, job, cfg):
                    skipped += 1
                    event("SKIP", index=index)
                    continue
                with _lock(root / "locks" / "state.lock"):
                    (root / "completed.json").unlink(missing_ok=True)
                video, meta = _job_paths(root, job)
                fd, name = tempfile.mkstemp(prefix=f".{index:04d}.", suffix=".tmp.mp4", dir=root / "videos")
                os.close(fd)
                temporary = Path(name)
                start = time.monotonic()
                try:
                    event("START_VIDEO", index=index, prompt=job["prompt"])
                    command = runner.generate(job, temporary)
                    s = cfg["settings"]
                    if not probe(temporary, s["frames"], s["width"], s["height"], s["fps"]):
                        raise RuntimeError(f"Output validation failed for index {index}")
                    video_hash = sha256(temporary)
                    with temporary.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, video)
                    _fsync_directory(video.parent)
                    elapsed = time.monotonic() - start
                    atomic_json(
                        meta,
                        dict(
                            job=job,
                            fingerprint=fingerprint(cfg, job),
                            configuration_sha256=fingerprint(cfg),
                            video_sha256=video_hash,
                            size_bytes=video.stat().st_size,
                            gpu=runner.gpu,
                            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            elapsed_seconds=elapsed,
                            finished_at=_now(),
                            command=command,
                        ),
                    )
                    count += 1
                    event("DONE_VIDEO", index=index, elapsed_seconds=elapsed, video_sha256=video_hash)
                finally:
                    temporary.unlink(missing_ok=True)
            if args.limit and count >= args.limit:
                break
        event(
            "WORKER_FINISHED",
            newly_generated=count,
            skipped=skipped,
            limited=bool(args.limit and count >= args.limit),
            note="Worker exit is not full-run completion; run verify",
        )
    except BaseException as exc:
        event("WORKER_FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            runner.close()
        finally:
            if old_attention is None:
                os.environ.pop("DIFFUSION_ATTENTION_BACKEND", None)
            else:
                os.environ["DIFFUSION_ATTENTION_BACKEND"] = old_attention


def verify(root, task):
    """Verify ALL expected jobs, metadata, video hashes and decoded formats.

    Return the completion manifest (including each video hash); atomically write
    completed.json only on success. No model loading or CUDA import is performed.
    """
    root = Path(root).expanduser().resolve()
    # Remove stale completion evidence even if the input/config checks now fail.
    with _lock(root / "locks" / "state.lock"):
        (root / "completed.json").unlink(missing_ok=True)
    task_record, jobs = read_prepared(root, task)
    config = load_json(root / "configuration.json")
    _check_configuration(config, task, task_record)
    failures = []
    videos = []
    # Holding the job locks through publication prevents concurrent repair from
    # racing the verified hash manifest. Workers use the same job -> state order.
    with contextlib.ExitStack() as stack:
        for job in jobs:
            stack.enter_context(_lock(root / "locks" / f"{job['index']:04d}.lock"))
        for job in jobs:
            if not _completed_job(root, job, config):
                failures.append(job["index"])
            else:
                _, metadata = _job_paths(root, job)
                record = load_json(metadata)
                videos.append(
                    dict(
                        index=job["index"],
                        filename=job["filename"],
                        fingerprint=record["fingerprint"],
                        video_sha256=record["video_sha256"],
                        size_bytes=record["size_bytes"],
                    )
                )
        if failures:
            raise RuntimeError(
                f"Verification failed: {len(failures)}/{len(jobs)} missing/invalid "
                f"jobs; indices={failures}. No full completion recorded."
            )
        manifest = dict(
            schema_version=1,
            task=task,
            selected_count=EXPECTED_COUNTS[task],
            videos_count=len(videos),
            task_sha256=fingerprint(task_record),
            configuration_sha256=fingerprint(config),
            jobs_sha256=fingerprint(jobs),
            videos=videos,
            videos_sha256=fingerprint(videos),
            verified_at=_now(),
        )
        with _lock(root / "locks" / "state.lock"):
            # Recheck immutable files before certifying the snapshot.
            if load_json(root / "configuration.json") != config or load_json(root / "jobs.json") != jobs:
                raise RuntimeError("Configuration/jobs changed during verification")
            atomic_json(root / "completed.json", manifest)
    print(f"VERIFIED {task} {len(videos)}/{EXPECTED_COUNTS[task]} videos", flush=True)
    return manifest


def _command_evidence(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return dict(
            command=command, returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip()
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(command=command, error=type(exc).__name__)


def environment(root=None, task=None, source=None):
    """Collect only allowlisted environment fields; never dump argv or os.environ."""
    source_info = validate_source(source) if source else None
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Isolate runtime imports from this stdlib CLI. No model/engine is constructed.
    code = """
import importlib.util, json, sys
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
origins = {}
for name in ('torch', 'numpy', 'vllm', 'vllm_omni', 'diffusers', 'flashinfer', 'cutlass', 'vbench'):
    try:
        spec = importlib.util.find_spec(name)
        origins[name] = spec.origin if spec else None
    except Exception as exc:
        origins[name] = {'error_type': type(exc).__name__}
result = {'imports': origins}
try:
    import torch
    result['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
                       'cuda_available': torch.cuda.is_available(), 'file': torch.__file__}
except Exception as exc:
    result['torch'] = {'error_type': type(exc).__name__}
print(json.dumps(result))
"""
    imports = _command_evidence([sys.executable, "-c", code] + ([str(Path(source).resolve())] if source else []))
    try:
        runtime = json.loads(imports.get("stdout", "").splitlines()[-1])
    except (ValueError, IndexError):
        runtime = dict(error="Runtime import probe failed", returncode=imports.get("returncode"))
    if source_info:
        origin = runtime.get("imports", {}).get("vllm_omni")
        if not isinstance(origin, str) or not Path(origin).resolve().is_relative_to(
            Path(source_info["path"]) / "vllm_omni"
        ):
            raise RuntimeError("Environment import probe did not resolve the selected Omni checkout")
    evidence = dict(
        schema_version=1,
        captured_at=_now(),
        task=task,
        python=dict(executable=sys.executable, version=sys.version),
        platform=platform.platform(),
        versions=versions,
        runtime=runtime,
        source=source_info,
        environment={
            key: os.environ.get(key)
            for key in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "DIFFUSION_ATTENTION_BACKEND")
        },
        commands={
            name: _command_evidence(command)
            for name, command in (
                ("nvcc", ["nvcc", "--version"]),
                ("nvidia-smi", ["nvidia-smi"]),
                ("ffmpeg", ["ffmpeg", "-version"]),
                ("ffprobe", ["ffprobe", "-version"]),
            )
        },
    )
    if root is not None:
        atomic_json(Path(root).expanduser().resolve() / "environment.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    return evidence


def evaluate(root, task, metrics=None):
    """Dispatch lazily to the companion evaluator; never score an unverified root."""
    selected = list(METRICS[_task_name(task)] if metrics is None else metrics)
    if not selected or len(set(selected)) != len(selected) or any(m not in METRICS[task] for m in selected):
        raise ValueError(f"Metrics must be distinct members of {METRICS[task]}")
    root = Path(root).expanduser().resolve()
    # Keep one module identity when the companion imports benchmark from a CLI run.
    sys.modules.setdefault("benchmark", sys.modules[__name__])
    path = Path(__file__).resolve().with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("_wan_benchmark_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Companion evaluator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate(root, task, selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "generate", "verify", "evaluate", "environment"):
        command = commands.add_parser(name)
        command.add_argument("--task", choices=tuple(SETTINGS), required=name != "environment")
        command.add_argument("--root", type=Path, required=name != "environment")
        if name == "prepare":
            command.add_argument("--i2v-images", type=Path)
        if name in ("generate", "environment"):
            command.add_argument("--omni-source", type=Path, required=name == "generate")
        if name == "generate":
            command.add_argument("--model", type=Path, required=True, help="Local immutable checkpoint directory")
            command.add_argument("--precision", choices=("bf16", "nvfp4", "mxfp8", "nvfp4_e5m3"), required=True)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
            command.add_argument("--limit", type=int, default=0, help="Maximum NEW videos per worker; 0 means no limit")
        if name == "evaluate":
            command.add_argument("--metrics", nargs="+", help="Task metric subset (default: all five)")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.root, args.task, args.i2v_images)
        elif args.command == "generate":
            generate(args)
        elif args.command == "verify":
            verify(args.root, args.task)
        elif args.command == "evaluate":
            evaluate(args.root, args.task, args.metrics)
        else:
            environment(args.root, args.task, args.omni_source)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

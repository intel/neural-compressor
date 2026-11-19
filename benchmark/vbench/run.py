#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Wan2.2 videos or evaluate a completed generation run."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from benchmark import METRICS, MODEL_CLASSES, bind_checkpoint, load_json, prepare

BENCHMARK = Path(__file__).with_name("benchmark.py")
PRECISIONS = {None: "bf16", "mx_fp": "mxfp8", "nv_fp": "nvfp4", "nvfp4_v2": "nvfp4_e5m3"}


def run_workers(commands, root):
    """Run stages in private process groups and clean them up on cancellation."""
    workers = []
    interrupted = None

    def cancel(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    def signal_worker(worker, signum):
        try:
            os.killpg(worker.pid, signum)
        except ProcessLookupError:
            pass

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    handlers = {signum: signal.signal(signum, cancel) for signum in signals}
    try:
        for command, gpu, log in commands:
            if interrupted is not None:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            with log.open("wb") if log else nullcontext() as output:
                worker = subprocess.Popen(
                    command, env=env, start_new_session=True, stdout=output, stderr=subprocess.STDOUT
                )
            workers.append(worker)
        while interrupted is None:
            statuses = [worker.poll() for worker in workers]
            if any(status for status in statuses):
                raise RuntimeError(f"Benchmark stage failed. See {root / 'logs'} for generation logs.")
            if all(status == 0 for status in statuses):
                return
            time.sleep(0.25)
        raise SystemExit(128 + interrupted)
    finally:
        active = [worker for worker in workers if worker.poll() is None]
        for worker in active:
            signal_worker(worker, signal.SIGTERM)
        for worker in active:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_worker(worker, signal.SIGKILL)
                worker.wait()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def generate_videos(args, output_dir, gpus):
    """Prepare inputs, check the checkpoint, and start one shard per GPU."""
    if args.task == "i2v" and args.i2v_images is None:
        raise ValueError("Set I2V_IMAGES or pass --i2v_images pointing to crop/16-9")

    model = Path(args.model_path).expanduser()
    if not model.is_dir():
        from huggingface_hub import snapshot_download

        model = Path(snapshot_download(args.model_path, revision=args.revision))
    model = model.resolve()
    model_class = MODEL_CLASSES[args.task]
    if load_json(model / "model_index.json").get("_class_name") != model_class:
        raise ValueError(f"--task={args.task} requires a {model_class} checkpoint")

    quant_config = model / "transformer" / "quantization_config.json"
    data_type = load_json(quant_config).get("data_type") if quant_config.exists() else None
    if data_type not in PRECISIONS:
        raise ValueError(f"Unsupported checkpoint data_type: {data_type}")
    precision = PRECISIONS[data_type]
    if args.dtype is not None and args.dtype != precision:
        raise ValueError(f"--dtype={args.dtype} does not match the checkpoint precision ({precision})")
    prepare(output_dir, args.task, args.i2v_images if args.task == "i2v" else None)
    bind_checkpoint(output_dir, args.task, model, precision)

    command = [sys.executable, str(BENCHMARK), "generate", "--task", args.task, "--root", str(output_dir)]
    logs = output_dir / "logs"
    workers = []
    for shard_index, gpu in enumerate(gpus):
        shard_args = ["--shard-index", str(shard_index), "--num-shards", str(len(gpus))]
        workers.append((command + shard_args, gpu, logs / f"gpu-{gpu}.log"))
    print(f"Generating {args.task} videos in {output_dir / 'videos'}. Worker logs: {logs}", flush=True)
    run_workers(workers, output_dir)


def evaluate_videos(args, output_dir, gpus):
    """Score the videos already written to the output directory."""
    if load_json(output_dir / "task.json")["task"] != args.task:
        raise ValueError("--task must match the task used to generate this output directory")
    command = [sys.executable, str(BENCHMARK), "evaluate", "--task", args.task, "--root", str(output_dir)]
    if args.metrics:
        command += ["--metrics", *args.metrics.split(",")]
    run_workers([(command, gpus[0], None)], output_dir)
    print(f"Scores: {output_dir / 'evaluation' / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(required=True)
    generate = stages.add_parser("generate", help="Generate videos for the fixed VBench inputs")
    evaluate = stages.add_parser("evaluate", help="Verify and score a completed video set")
    generate.set_defaults(run_stage=generate_videos)
    evaluate.set_defaults(run_stage=evaluate_videos)
    for stage in (generate, evaluate):
        stage.add_argument("--task", choices=tuple(METRICS), required=True)
        stage.add_argument("--output_dir", type=Path, required=True)
        stage.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    generate.add_argument("--model_path", required=True, help="Local checkpoint or Hugging Face model ID")
    generate.add_argument(
        "--dtype",
        choices=tuple(PRECISIONS.values()),
        help="Check the expected precision (default: read the checkpoint)",
    )
    generate.add_argument("--revision", help="Hugging Face checkpoint revision")
    generate.add_argument("--i2v_images", type=Path, default=os.environ.get("I2V_IMAGES"))
    evaluate.add_argument("--metrics", help="Comma-separated metric subset")
    args = parser.parse_args()

    gpus = args.gpus.split(",")
    if any(not gpu.isascii() or not gpu.isdecimal() for gpu in gpus):
        parser.error("--gpus requires comma-separated GPU indices")
    gpus = [str(int(gpu)) for gpu in gpus]
    if len(set(gpus)) != len(gpus):
        parser.error("Duplicate GPU indices")
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the lock until the workers exit so scoring can't overlap generation in this directory.
        with (output_dir / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            args.run_stage(args, output_dir, gpus)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

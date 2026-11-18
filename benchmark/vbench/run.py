#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Wan2.2 videos or evaluate a completed generation run."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
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
            try:
                next(worker for worker in workers if worker.returncode is None).wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    generate = stages.add_parser("generate", help="Generate videos for the fixed VBench inputs")
    evaluate = stages.add_parser("evaluate", help="Verify and score a completed video set")
    for stage in (generate, evaluate):
        stage.add_argument("--task", choices=tuple(METRICS), required=True)
        stage.add_argument("--output_dir", type=Path, required=True)
        stage.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    generate.add_argument("--model_path", required=True, help="Local checkpoint or Hugging Face model ID")
    generate.add_argument("--revision", help="Hugging Face checkpoint revision")
    generate.add_argument("--omni_source", type=Path, default=os.environ.get("OMNI_SOURCE"))
    generate.add_argument("--i2v_images", type=Path, default=os.environ.get("I2V_IMAGES"))
    evaluate.add_argument("--metrics", help="Comma-separated metric subset")
    args = parser.parse_args()

    gpus = args.gpus.split(",")
    if any(not gpu.isascii() or not gpu.isdecimal() for gpu in gpus):
        parser.error("--gpus requires comma-separated GPU indices")
    gpus = [str(int(gpu)) for gpu in gpus]
    if len(set(gpus)) != len(gpus):
        parser.error("Duplicate GPU indices")
    root = args.output_dir.expanduser().resolve()
    base = [sys.executable, str(BENCHMARK), args.stage, "--task", args.task, "--root", str(root)]

    try:
        root.mkdir(parents=True, exist_ok=True)
        # Keep generation and evaluation from using the same outputs at the same time.
        with (root / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.stage == "generate":
                if args.omni_source is None:
                    parser.error("Set OMNI_SOURCE or pass --omni_source")
                if args.task == "i2v" and args.i2v_images is None:
                    parser.error("Set I2V_IMAGES or pass --i2v_images pointing to crop/16-9")
                model = Path(args.model_path).expanduser()
                if not model.is_dir():
                    from huggingface_hub import snapshot_download

                    model = Path(snapshot_download(args.model_path, revision=args.revision))
                model = model.resolve()
                if load_json(model / "model_index.json").get("_class_name") != MODEL_CLASSES[args.task]:
                    parser.error(f"--task={args.task} requires a {MODEL_CLASSES[args.task]} checkpoint")
                quant = model / "transformer" / "quantization_config.json"
                dtype = load_json(quant).get("data_type") if quant.exists() else None
                if dtype not in PRECISIONS:
                    parser.error(f"Unsupported checkpoint data_type: {dtype}")
                prepare(root, args.task, args.i2v_images if args.task == "i2v" else None)
                bind_checkpoint(root, args.task, model, PRECISIONS[dtype], args.omni_source)
                commands = [
                    (
                        base + ["--shard-index", str(index), "--num-shards", str(len(gpus))],
                        gpu,
                        root / "logs" / f"gpu-{gpu}.log",
                    )
                    for index, gpu in enumerate(gpus)
                ]
                print(f"Generating {args.task} videos in {root / 'videos'}. Worker logs: {root / 'logs'}", flush=True)
                run_workers(commands, root)
            else:
                if load_json(root / "task.json")["task"] != args.task:
                    parser.error("--task must match the task used to generate this output directory")
                if args.metrics:
                    base += ["--metrics", *args.metrics.split(",")]
                run_workers([(base, gpus[0], None)], root)
                print(f"Scores: {root / 'evaluation' / 'summary.json'}", flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

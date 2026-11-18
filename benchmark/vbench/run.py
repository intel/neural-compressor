#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-card entry point for the fixed Wan2.2 VBench protocols."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark import MODEL_CLASSES, validate_model, validate_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Hub revision; use a full commit for immutable downloads")
    parser.add_argument("--output_dir", type=Path, help="One run directory per model/task/precision")
    parser.add_argument(
        "--omni_source",
        type=Path,
        default=os.environ.get("OMNI_SOURCE"),
        help="Pinned Omni checkout (default: OMNI_SOURCE)",
    )
    parser.add_argument(
        "--i2v_images",
        type=Path,
        default=os.environ.get("I2V_IMAGES"),
        help="Official crop/16-9 directory (default: I2V_IMAGES)",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated physical GPU indices; one full model per GPU",
    )
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all")
    parser.add_argument("--metrics", help="Comma-separated metric subset for evaluation")
    args = parser.parse_args(argv)
    tokens = args.gpus.split(",")
    if any(not t.isascii() or not t.isdecimal() for t in tokens):
        parser.error("--gpus requires comma-separated nonnegative GPU indices")
    gpus = [str(int(t)) for t in tokens]
    if len(set(gpus)) != len(gpus):
        parser.error("Duplicate GPU indices")
    metrics = args.metrics.split(",") if args.metrics else None
    if args.stage == "evaluate":
        if args.output_dir is None:
            parser.error("--stage=evaluate requires --output_dir of an existing run")
        root = args.output_dir.expanduser().resolve()
        config = json.loads((root / "configuration.json").read_text())
        task = config["task"]
        model = config["model"]["path"]
        if args.model_path and Path(args.model_path).expanduser().resolve() != Path(model):
            parser.error("For evaluation, omit --model_path or pass the recorded local checkpoint path")
    else:
        if not args.model_path:
            parser.error("--model_path is required for generation")
        if args.omni_source is None:
            parser.error("Set OMNI_SOURCE or --omni_source to the pinned Omni checkout")
        validate_source(args.omni_source)
        model = Path(args.model_path).expanduser()
        if not model.is_dir():
            from huggingface_hub import snapshot_download

            model = Path(snapshot_download(args.model_path, revision=args.revision))
        model = model.resolve()
        index = json.loads((model / "model_index.json").read_text())
        tasks = {v: k for k, v in MODEL_CLASSES.items()}
        if index.get("_class_name") not in tasks:
            parser.error("Expected WanPipeline or WanImageToVideoPipeline")
        task = tasks[index["_class_name"]]
        quant_path = model / "transformer" / "quantization_config.json"
        dtype = json.loads(quant_path.read_text()).get("data_type") if quant_path.exists() else None
        precisions = {None: "bf16", "mx_fp": "mxfp8", "nv_fp": "nvfp4", "nvfp4_v2": "nvfp4_e5m3"}
        if dtype not in precisions:
            parser.error(f"Unsupported checkpoint data_type: {dtype}")
        precision = precisions[dtype]
        validate_model(model, task, precision)
        if task == "i2v" and args.i2v_images is None:
            parser.error("I2V requires I2V_IMAGES or --i2v_images pointing to crop/16-9")
        identity = hashlib.sha256(str(model).encode()).hexdigest()[:12]
        root = (args.output_dir or Path("runs") / f"{task}-{precision}-{identity}").expanduser().resolve()
    from benchmark import METRICS

    if metrics and (len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS[task])):
        parser.error(f'Choose distinct {task} metrics from {", ".join(METRICS[task])}')
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHON=sys.executable, PYTHONUNBUFFERED="1")
    print(f"Task: {task}; checkpoint: {model}; output: {root}", flush=True)
    if args.stage != "evaluate":
        command = [
            "bash",
            str(scripts / "launch_workers.sh"),
            task,
            str(model),
            precision,
            str(root),
            str(args.omni_source.expanduser().resolve()),
            ",".join(gpus),
        ]
        if task == "i2v":
            command.append(str(args.i2v_images.expanduser().resolve()))
        subprocess.run(command, env=env, check=True)
    stage = "verify" if args.stage == "generate" else "evaluate"
    command = [sys.executable, str(scripts / "benchmark.py"), stage, "--task", task, "--root", str(root)]
    if stage == "evaluate":
        if metrics:
            command += ["--metrics", *metrics]
    subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=gpus[0]), check=True)
    if stage == "evaluate":
        print(f'Scores: {root / "evaluation/summary.json"}', flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score completed vLLM-Omni runs with CLIP, CLIP-IQA, and ImageReward."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from benchmark import FIELDS, atomic_json, atomic_write, csv_bytes, load_manifest, open_run_lock, positive_int

CLIP_MODEL = "openai/clip-vit-base-patch32"
IQA_PAIRS = (
    ("Good photo.", "Bad photo."),
    ("Good quality.", "Bad quality."),
    ("Aesthetically pleasing.", "Aesthetically displeasing."),
)
METRICS = ("clip", "clip_iqa", "image_reward")
SUMMARY_FIELDS = (
    "model",
    "revision",
    "dtype",
    "dataset",
    "manifest_sha256",
    "steps",
    "guidance",
    "width",
    "height",
    "base_seed",
    "category",
    "count",
    *METRICS,
)


def open_rgb(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def clip_scores(rows, device, batch_size):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL).eval().to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    texts = [text for pair in IQA_PAIRS for text in pair]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [open_rgb(row["image_path"]) for row in batch]
        pixels = processor.image_processor(images, return_tensors="pt").pixel_values.to(device)
        captions = processor.tokenizer(
            [row["caption"] for row in batch], padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        iqa = processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            alignment = model(pixel_values=pixels, **captions).logits_per_image.diagonal()
            pair_logits = model(pixel_values=pixels, **iqa).logits_per_image.reshape(len(batch), len(IQA_PAIRS), 2)
            quality = pair_logits.softmax(dim=-1)[..., 0].mean(dim=-1) * 100
        for row, clip, iqa_value in zip(batch, alignment.tolist(), quality.tolist(), strict=True):
            row.update(clip=clip, clip_iqa=iqa_value)
        print(f"CLIP and CLIP-IQA: {start + len(batch)}/{len(rows)}", flush=True)
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def reward_worker(device, input_path, output_path):
    import ImageReward
    import torch

    rows = json.loads(Path(input_path).read_text())
    model = ImageReward.load("ImageReward-v1.0", device=device).eval()
    scores = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            scores.append(float(model.score(row["caption"], open_rgb(row["image_path"]))))
            if i % 100 == 0 or i == len(rows):
                print(f"ImageReward: {i}/{len(rows)}", flush=True)
    atomic_json(Path(output_path), scores)


def reward_scores(rows, args):
    env = os.environ.copy()
    if args.imagereward_env:
        directory = args.imagereward_env.expanduser().resolve()
        if not (directory / "ImageReward").is_dir():
            raise FileNotFoundError(f"ImageReward missing from isolated directory {directory}")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(directory), env.get("PYTHONPATH"))))
    with TemporaryDirectory() as work:
        inputs = Path(work) / "inputs.json"
        output = Path(work) / "scores.json"
        atomic_json(inputs, [{"caption": row["caption"], "image_path": row["image_path"]} for row in rows])
        subprocess.run(
            [args.reward_python, str(Path(__file__).resolve()), "_reward", args.device, str(inputs), str(output)],
            env=env,
            check=True,
        )
        scores = json.loads(output.read_text())
    for row, value in zip(rows, scores, strict=True):
        row["image_reward"] = value


def score_run(run, args):
    with ExitStack() as lock:
        open_run_lock(run, lock)
        config = json.loads((run / "run.json").read_text())
        rows, data = load_manifest(run / "prompts.tsv")
        if hashlib.sha256(data).hexdigest() != config["manifest_sha256"]:
            raise ValueError(f"Prompt manifest changed for {run}")

        from PIL import Image

        for row in rows:
            path = (run / "images" / f"{row['id']}.png").resolve()
            with Image.open(path) as image:
                if image.size != (config["width"], config["height"]):
                    raise ValueError(f"Unexpected resolution for {path}")
                image.verify()
            row["image_path"] = str(path)
            row["seed"] = config["base_seed"] + row["id"]

        clip_scores(rows, args.device, args.clip_batch_size)
        reward_scores(rows, args)
        fields = [key for key in SUMMARY_FIELDS if key in config]
        overall = {key: config[key] for key in fields}
        groups = defaultdict(list)
        for row in rows:
            if row["category"]:
                groups[row["category"]].append(row)

        def summary(category, items):
            return overall | {
                "category": category,
                "count": len(items),
                **{metric: fmean(row[metric] for row in items) for metric in METRICS},
            }

        aggregate = summary("", rows)
        summaries = [aggregate] + [summary(category, items) for category, items in sorted(groups.items())]
        atomic_write(run / "per_image_scores.csv", csv_bytes(rows, (*FIELDS, "seed", "image_path", *METRICS)))
        atomic_write(run / "summary.csv", csv_bytes(summaries, SUMMARY_FIELDS))
        atomic_json(
            run / "summary.json",
            {
                "clip_model": CLIP_MODEL,
                "clip_iqa_pairs": IQA_PAIRS,
                "image_reward_model": "ImageReward-v1.0",
                "overall": aggregate,
                "categories": summaries[1:],
            },
        )
        print(json.dumps(aggregate, ensure_ascii=False), flush=True)
        return aggregate


def find_runs(output_dir, datasets=None):
    output_dir = output_dir.expanduser()
    runs = (
        [output_dir / dataset for dataset in datasets]
        if datasets
        else sorted(path.parent for path in output_dir.glob("*/run.json"))
    )
    if not runs:
        raise ValueError(f"No generated datasets in {output_dir}")
    for run in runs:
        if not (run / "run.json").is_file():
            raise ValueError(f"No generated dataset in {run}")
    return runs


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "_reward":
        reward_worker(*sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="score images in a generated output directory")
    evaluate.add_argument("--output_dir", type=Path, required=True)
    evaluate.add_argument("--dataset", choices=("drawbench", "coco2014"), action="append")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--clip_batch_size", type=positive_int, default=16)
    evaluate.add_argument("--reward_python", default=sys.executable)
    evaluate.add_argument("--imagereward_env", type=Path)
    summarize = commands.add_parser("summarize", help="combine existing summaries without rescoring")
    summarize.add_argument("runs", type=Path, nargs="+", help="generation output directories")
    summarize.add_argument("--output", type=Path, required=True, help="combined summary CSV")
    args = parser.parse_args()
    if args.command == "evaluate":
        for run in find_runs(args.output_dir, args.dataset):
            score_run(run, args)
    else:
        summaries = [
            json.loads((run / "summary.json").read_text())["overall"]
            for output_dir in args.runs
            for run in find_runs(output_dir)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.output, csv_bytes(summaries, SUMMARY_FIELDS))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DrawBench generation + scoring for Z-Image-Turbo (aqa-compatible).

This example mirrors ``autoquant-agent``'s ``drawbench_metrics`` recipe:

* generation: vllm-omni (same backend/params/seeds as aqa's diffusion harness)
  or diffusers (bf16 only);
* scoring: CLIP (``openai/clip-vit-base-patch32``, logits_per_image x100),
  CLIP-IQA (3 antonym prompt pairs averaged, x100) and ImageReward-v1.0.

Those metric definitions are intentionally different from ``flux/main.py``
(ViT-L/14 + single ``quality`` prompt), so numbers match aqa/model cards
instead of the torchmetrics defaults.

Examples::

    # 1) build the TSV once
    python prepare_drawbench.py --output drawbench.tsv

    # 2) generate + score (vllm-omni, MXFP8 checkpoint)
    python main.py --inference --accuracy \
        --model /path/to/Z-Image-Turbo-MXFP8-RTN-AutoRound \
        --eval_dataset drawbench.tsv --output_image_path ./images

    # 3) score already generated images (names must be <id>.png)
    python main.py --accuracy --eval_dataset drawbench.tsv \
        --output_image_path ./images
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tabulate
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser(
    description="Z-Image-Turbo DrawBench generation/eval (aqa-compatible).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", "--model_name_or_path", help="model directory (bf16 or auto-round checkpoint)")
parser.add_argument("--eval_dataset", default="drawbench.tsv", help="TSV with id and caption columns")
parser.add_argument("--output_image_path", default="./tmp_imgs", help="directory for <id>.png images")
parser.add_argument("--inference", action="store_true", help="generate images")
parser.add_argument("--accuracy", action="store_true", help="score images")
parser.add_argument("--backend", default="vllm_omni", choices=["vllm_omni", "diffusers"],
                    help="generation backend; vllm_omni matches aqa's diffusion harness")
parser.add_argument("--num_inference_steps", default=8, type=int)
parser.add_argument("--guidance_scale", default=0.0, type=float)
parser.add_argument("--height", default=1024, type=int)
parser.add_argument("--width", default=1024, type=int)
parser.add_argument("--seed", default=42, type=int, help="image i uses seed + i (aqa convention)")
parser.add_argument("--limit", default=-1, type=int)
parser.add_argument("--metrics", default="clip,clip-iqa,imagereward")
parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
parser.add_argument("--imagereward_env", default=os.environ.get(
    "AQA_IMAGEREWARD_ENV", "~/.cache/aqa/envs/imagereward"),
    help="isolated transformers-4 dir for ImageReward")
parser.add_argument("--overwrite", action="store_true", help="regenerate existing images")
args = parser.parse_args()

# CLIP-IQA antonym prompt pairs (same as aqa's drawbench_metrics).
CLIP_IQA_PROMPT_PAIRS = [
    ("Good photo.", "Bad photo."),
    ("Good quality.", "Bad quality."),
    ("Aesthetically pleasing.", "Aesthetically displeasing."),
]

IMAGEREWARD_WORKER = r"""
import json, sys
from PIL import Image

payload = json.loads(sys.argv[1])
import ImageReward
rm = ImageReward.load(payload["model"])
out = [float(rm.score(r["prompt"], [Image.open(r["image_path"])])) for r in payload["rows"]]
print(json.dumps(out))
"""


def load_rows(path):
    df = pd.read_csv(path, sep="\t")
    for col in ("id", "caption"):
        if col not in df.columns:
            raise ValueError(f"{path} must contain '{col}' column")
    if args.limit > 0:
        df = df.iloc[: args.limit]
    return df.reset_index(drop=True)


def image_path(out_dir, row_id):
    return os.path.join(out_dir, f"{row_id}.png")


# ── generation ───────────────────────────────────────────────────────────────
def _omni_sampling_params(omni, seed):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=1,
    )
    defaults = list(getattr(omni, "default_sampling_params_list", None) or [])
    if not defaults:
        return [params]
    resolved, replaced = [], False
    for default in defaults:
        if isinstance(default, OmniDiffusionSamplingParams):
            resolved.append(params)
            replaced = True
        else:
            resolved.append(default)
    if not replaced and len(resolved) == 1:
        resolved = [params]
    return resolved


def _first_image(outputs):
    for output in outputs or []:
        images = getattr(output, "images", None)
        if images:
            return images[0]
    raise RuntimeError("vllm-omni returned no images")


def generate_vllm_omni(rows, out_dir):
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=args.model, mode="text-to-image")
    for idx, row in rows.iterrows():
        path = image_path(out_dir, row["id"])
        if os.path.exists(path) and not args.overwrite:
            continue
        outputs = omni.generate(
            {"prompt": str(row["caption"]), "modalities": ["image"]},
            sampling_params_list=_omni_sampling_params(omni, args.seed + idx),
        )
        _first_image(outputs).save(path)
        print(f"generated {path} (seed={args.seed + idx})", flush=True)


def generate_diffusers(rows, out_dir):
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device)
    for idx, row in rows.iterrows():
        path = image_path(out_dir, row["id"])
        if os.path.exists(path) and not args.overwrite:
            continue
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        out = pipe(
            prompt=str(row["caption"]),
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        out.images[0].save(path)
        print(f"generated {path} (seed={args.seed + idx})", flush=True)


# ── scoring (aqa-compatible definitions) ─────────────────────────────────────
def clip_scores(prompts, images, device):
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    scores = []
    for prompt, img_path in zip(prompts, images):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=[prompt], images=[image], return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            scores.append(float(model(**inputs).logits_per_image[0, 0]))
    return scores


def clip_iqa_scores(images, device):
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    scores = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        pair_scores = []
        for positive, negative in CLIP_IQA_PROMPT_PAIRS:
            inputs = processor(text=[positive, negative], images=[image, image],
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[[0, 1], [0, 1]]
            pair_scores.append(float(logits.softmax(dim=0)[0]) * 100.0)
        scores.append(sum(pair_scores) / len(pair_scores))
    return scores


def _imagereward_direct(rows):
    import ImageReward

    model = ImageReward.load("ImageReward-v1.0")
    return [float(model.score(r["prompt"], [Image.open(r["image_path"])])) for r in rows]


def _imagereward_subprocess(rows):
    env_dir = Path(os.path.expanduser(args.imagereward_env))
    if not (env_dir / "ImageReward").is_dir():
        raise RuntimeError(
            f"ImageReward is not importable and no isolated env at {env_dir}. "
            "Install it with: pip install --target <dir> 'transformers==4.44.2' "
            "'diffusers==0.29.2' image-reward")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(env_dir)
    payload = json.dumps({"model": "ImageReward-v1.0", "rows": rows})
    proc = subprocess.run([sys.executable, "-c", IMAGEREWARD_WORKER, payload],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"ImageReward subprocess failed:\n{proc.stderr[-2000:]}")
    # ImageReward prints "load checkpoint ..." to stdout before the JSON line.
    json_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("[")]
    if not json_lines:
        raise RuntimeError(f"ImageReward subprocess produced no JSON:\n{proc.stdout[-2000:]}")
    return json.loads(json_lines[-1])


def imagereward_scores(rows):
    try:
        return _imagereward_direct(rows)
    except Exception as exc:  # transformers-5 incompatibility -> isolated env
        print(f"direct ImageReward import failed ({exc}); using isolated env", flush=True)
        return _imagereward_subprocess(rows)


def main():
    rows = load_rows(args.eval_dataset)
    out_dir = Path(args.output_image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.inference:
        if not args.model:
            raise ValueError("--inference needs --model")
        if args.backend == "vllm_omni":
            generate_vllm_omni(rows, out_dir)
        else:
            generate_diffusers(rows, out_dir)

    if args.accuracy:
        prompts, images = [], []
        for _, row in rows.iterrows():
            path = image_path(out_dir, row["id"])
            if os.path.exists(path):
                prompts.append(str(row["caption"]))
                images.append(path)
        if not images:
            raise RuntimeError(f"no images found under {out_dir}")

        wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
        result = {}
        if "clip" in wanted:
            result["CLIP"] = float(np.mean(clip_scores(prompts, images, device)))
        if "clip-iqa" in wanted:
            result["CLIP-IQA"] = float(np.mean(clip_iqa_scores(images, device)))
        if "imagereward" in wanted:
            irows = [{"prompt": p, "image_path": i} for p, i in zip(prompts, images)]
            result["ImageReward"] = float(np.mean(imagereward_scores(irows)))

        print(f"scored {len(images)} images with {args.clip_model}")
        print(tabulate.tabulate(result.items(), tablefmt="grid", floatfmt=".4f"))


if __name__ == "__main__":
    main()

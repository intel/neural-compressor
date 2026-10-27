# Z-Image-Turbo DrawBench evaluation (aqa-compatible)

This example generates DrawBench images with **vllm-omni** and scores them with
the same metric definitions used by `autoquant-agent` (aqa), so its numbers can
be compared directly with aqa runs and the published model cards.

## Metrics (why they differ from `flux/main.py`)

| Metric | This example / aqa | `flux/main.py` |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32`, `logits_per_image` x100 | `clip-vit-large-patch14`, x100 |
| CLIP-IQA | 3 antonym pairs averaged, softmax(pos, neg) x100 | torchmetrics default: single `quality` pair, 0..1 |
| ImageReward | ImageReward-v1.0 | ImageReward-v1.0 |

Because of the backbone/prompt differences, the absolute values are not
comparable across the two examples. With matched backbone + prompt set the two
implementations agree within 0.4 points.

## Prerequisites

- A Python env with `torch`, `transformers`, `diffusers`, `datasets`, `pandas`,
  `tabulate`, `Pillow` and (for generation) a working **vllm-omni** install.
- ImageReward: `image-reward==1.5` is incompatible with transformers 5.x. Either
  run the scoring in an env with `transformers<4.45`, or provision the isolated
  directory used by aqa:

  ```bash
  pip install --target ~/.cache/aqa/envs/imagereward \
      "transformers==4.44.2" "diffusers==0.29.2" image-reward \
      "clip @ https://github.com/openai/CLIP/archive/refs/heads/main.zip"
  # remove torch/torchvision/nvidia* from that dir so the host torch is used
  ```

  `main.py` auto-detects this dir (`AQA_IMAGEREWARD_ENV` overrides it) and runs
  ImageReward in a subprocess.

## Prepare DrawBench TSV

```bash
python prepare_drawbench.py --output drawbench.tsv
```

The TSV has `id` (0..199) and `caption`; image `i` is generated with
`seed + i` (seed defaults to 42), matching aqa's diffusion harness.

## Generate + score (vllm-omni, MXFP8 checkpoint)

```bash
python main.py --inference --accuracy \
    --model /path/to/Z-Image-Turbo-MXFP8-RTN-AutoRound \
    --eval_dataset drawbench.tsv \
    --output_image_path ./drawbench_images
```

- `--backend vllm_omni` (default) reproduces aqa's generation exactly:
  8 steps, guidance 0.0, 1024x1024, seed 42+i.
- `--backend diffusers` loads the checkpoint with `AutoPipelineForText2Image`
  (bf16 only; auto-round MXFP8 checkpoints need vllm-omni).

## Score existing images only

Images must be named `<id>.png` and match the TSV ids:

```bash
python main.py --accuracy \
    --eval_dataset drawbench.tsv \
    --output_image_path ./drawbench_images
```

## Expected results (published MXFP8 checkpoint)

With `--clip_model openai/clip-vit-base-patch32` and the 3-pair CLIP-IQA:

```
+-------------+---------+
| CLIP        | 31.7738 |
| CLIP-IQA    | 70.5770 |
| ImageReward |  0.9825 |
+-------------+---------+
```

These match aqa's `drawbench_metrics` run on
`INCModel3/Z-Image-Turbo-MXFP8-RTN-AutoRound` (31.774 / 70.577 / 0.9825).

## Multi-GPU

`run_benchmark.sh` follows the `flux` example: it splits the TSV per visible
GPU, runs one process per GPU writing into the same image directory, then
scores the full set.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh \
    --topology=zimage_mxfp8 \
    --input_model=/path/to/Z-Image-Turbo-MXFP8-RTN-AutoRound \
    --dataset_location=drawbench.tsv \
    --output_image_path=./drawbench_images
```

## Reproduction check (verified 2026-10-26)

Command used (single B200, `.venv-omini-latest`):

```bash
python main.py --inference --accuracy \
    --model /home/kaokaolv/models/Z-Image-Turbo-MXFP8-RTN-AutoRound \
    --eval_dataset drawbench.tsv \
    --output_image_path ./drawbench_images
```

Result vs autoquant-agent (aqa) on the same published checkpoint:

| Item | aqa | this example |
|---|---|---|
| CLIP | 31.7738 | **31.7738** |
| CLIP-IQA | 70.5770 | **70.5770** |
| ImageReward | 0.9825 | **0.9825** |
| images | 200 | 200 (all 200 pixel-identical to aqa) |

Notes:

- The full 200-image run is bitwise identical to aqa's generation in the same
  environment, and the metric code is the same, so the scores match exactly.
- vllm-omni is not guaranteed to be bitwise deterministic across environments or
  cold starts: two early short `--limit` runs with the same seed produced
  different pixels, while later runs (and the full run) matched. For strict
  comparison use the full TSV or score the same image directory.
- `--clip_model` defaults to `openai/clip-vit-base-patch32` and CLIP-IQA uses the
  3-pair average, which is what aqa/model cards report. Using the `flux/main.py`
  defaults (ViT-L/14, single `quality` prompt) gives different absolute values.

# Text-to-image benchmark

Generate images with vLLM-Omni and score DrawBench and COCO2014 captions with
CLIP, CLIP-IQA, and ImageReward. The runner supports unquantized BF16 baselines
and AutoRound MXFP8 checkpoints for FLUX and Z-Image.

## Environment setup

Use Linux x86-64, Python 3.12, a CUDA 13 compatible driver, and `uv` on `PATH`.
The GPU smoke tests use NVIDIA B200. From the repository root:

```bash
bash benchmark/t2i/setup.sh
```

This creates `venv/t2i` for vLLM-Omni and `venv/t2i-eval` for scoring. The
scoring environment uses Transformers 4.x for ImageReward. Set
`T2I_GENERATE_PYTHON` or `T2I_EVALUATE_PYTHON` to use an existing environment.
Set `HF_TOKEN` in your environment for Hugging Face checkpoints that require it.

## Prepare prompts

```bash
bash benchmark/t2i/run_t2i.sh prepare drawbench --output runs/t2i/prompts/drawbench.tsv
bash benchmark/t2i/run_t2i.sh prepare coco2014 --source /path/to/captions_val2014.json \
  --output runs/t2i/prompts/coco2014.tsv
```

- **DrawBench** uses 200 prompts from a pinned revision of
  [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench).
- **COCO2014** samples 5,000 distinct `val2014` image IDs with selection seed 42.
  It uses the caption with the lowest annotation ID for each selected image.
  `--source` accepts the [caption JSON](https://github.com/tylin/coco-caption/blob/master/annotations/captions_val2014.json)
  or the `annotations_trainval2014.zip` archive from [COCO](https://cocodataset.org/#download).

Add `--count 2` to each preparation command for a smoke test. Reuse the same
TSVs across checkpoints. With the default `--seed=42`, manifest ID `i` uses
generation seed `42 + i`.

## Generate and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash benchmark/t2i/run_t2i.sh generate \
  --model_path=Tongyi-MAI/Z-Image-Turbo --dtype=bf16 --output_dir=runs/t2i/z-image-bf16 \
  --manifest drawbench runs/t2i/prompts/drawbench.tsv \
  --manifest coco2014 runs/t2i/prompts/coco2014.tsv

CUDA_VISIBLE_DEVICES=0 bash benchmark/t2i/run_t2i.sh generate \
  --model_path=intel-ai/Z-Image-Turbo-MXFP8-AutoRound --dtype=mxfp8 --output_dir=runs/t2i/z-image-mxfp8 \
  --manifest drawbench runs/t2i/prompts/drawbench.tsv \
  --manifest coco2014 runs/t2i/prompts/coco2014.tsv

for precision in bf16 mxfp8; do
  CUDA_VISIBLE_DEVICES=0 bash benchmark/t2i/run_t2i.sh evaluate --output_dir="runs/t2i/z-image-$precision"
done
bash benchmark/t2i/run_t2i.sh summarize --output runs/t2i/summary.csv runs/t2i/z-image-{bf16,mxfp8}
```

`--model_path` accepts a local Diffusers-format checkpoint or a Hugging Face ID.
Omit `--dtype` to infer the precision from checkpoint metadata. `bf16` validates
an unquantized checkpoint and uses BF16 inference. `mxfp8` validates AutoRound
MXFP8 metadata. Hub runs record the resolved revision.

The default resolution is 1024 × 1024. For local paths with a generic name,
set `--family` or pass both `--steps` and `--guidance`.

| `--family` | Steps | Guidance |
| --- | ---: | ---: |
| `flux1-dev` | 20 | 3.5 |
| `flux1-schnell` | 4 | 0.0 |
| `flux2-dev` (including Multimodal) | 50 | 4.0 |
| `flux2-klein` | 4 | 1.0 |
| `z-image-turbo` | 8 | 0.0 |

For FLUX.2-dev, `--engine-kwargs '{"enable_cpu_offload":true}'` enables CPU
offload. The runner sets a longer initialization timeout for its larger weights.
Multimodal MXFP8 configures the transformer and text encoder separately according
to the checkpoint's quantization metadata.

Each `<output_dir>/<dataset>/` contains `images/`, `prompts.tsv`, and `run.json`.
Rerun generation with the same configuration to fill missing or damaged images.
Evaluation writes `per_image_scores.csv`, `summary.csv`, and `summary.json`.
DrawBench summaries include per-category scores.

## Metrics

- **CLIP** is the paired image-text logit from `openai/clip-vit-base-patch32`.
- **CLIP-IQA** averages the positive probabilities for three pairs (good/bad
  photo, good/bad quality, and aesthetically pleasing/displeasing), then scales
  the result to 0–100.
- **ImageReward** scores each generated image against its prompt with
  `ImageReward-v1.0`.

Overall scores are per-image means. Compare runs with the same manifest,
resolution, steps, guidance, and seed.

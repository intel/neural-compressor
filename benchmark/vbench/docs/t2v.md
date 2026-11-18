# Text-to-video protocol

[Back to overview](../README.md) · [Environment](environment.md) · [Recorded results](../results/t2v.json)

## Inputs and generation

[The fixed prompt snapshot](../data/t2v_prompts.json) contains **165 jobs in historical order**: 72 from `subject_consistency`, then 93 from `overall_consistency`. These names identify prompt sources, not exclusive evaluation assignments. Do not sort, deduplicate, replace prompts, or average separate group scores.

| Setting | Recorded value |
|---|---|
| Resolution | 1280 × 720 |
| Frames / export rate | 81 / 16 fps |
| Inference steps | 40 |
| Seed | 42 for each input |
| Guidance scales | 4.0 / 3.0 |
| Boundary ratio | 0.875 |
| Flow shift | 5.0 |
| Negative prompt | Official Wan negative prompt, retained verbatim in the [protocol JSON](../results/t2v.json) |
| Attention | `TRTLLM_ATTN` |
| Linear backend | Explicit `flashinfer_cutlass` |
| Execution | Eager; no diffusion cache |

These are the measured comparison settings, not a claim that every value is an upstream model-card default. In particular, retain flow shift 5.0 for this comparison.

Generation calls the official `Omni` API with `OmniDiffusionSamplingParams`, using [the pinned source](https://github.com/vllm-project/vllm-omni/tree/7465c0f52bcb46a55f62c1b8613846cda30d0083). The T2V path explicitly requests `linear_backend="flashinfer_cutlass"`; it is not the unchanged I2V example path. Load the original local checkpoint directly—do not rewrite quantization metadata or create a synthetic runtime checkpoint view. See [model IDs and export provenance](quantization.md).

## Generate, resume, and verify

```bash
export OMNI_SOURCE=/path/to/vllm-omni
export MODEL=/path/to/local/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound
python scripts/benchmark.py prepare --task t2v --root runs/t2v-mxfp8
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py generate \
  --task t2v --root runs/t2v-mxfp8 --model "$MODEL" --precision mxfp8 \
  --omni-source "$OMNI_SOURCE" --shard-index 0 --num-shards 1
python scripts/benchmark.py verify --task t2v --root runs/t2v-mxfp8
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task t2v --root runs/t2v-mxfp8
```

Preparation stores the immutable 165-job input snapshot only. Generation persists model/precision/source/settings; do not reuse that root for BF16 or NVFP4. A matching fingerprint and `ffprobe` validation are required for resume. Missing, mismatched, or invalid videos are not successful completions.

For a separate smoke run, prepare a different root and add `--limit 1` to generation. This leaves 165 expected jobs; it is not a one-video benchmark, and full evaluation must reject an incomplete root.

### Multiple GPUs

```bash
bash scripts/launch_workers.sh t2v "$MODEL" mxfp8 runs/t2v-mxfp8 \
  "$OMNI_SOURCE" 0,1,2
```

The portable CLI uses deterministic indexed shards, with `jobs[index::num_shards]` input ownership. Use any suitable GPU list and one complete model per worker. The historical T2V scheduler instead used dynamic per-job `flock` claims; the portable launcher reproduces the input/configuration protocol, not that scheduling policy or its timings. Do not run overlapping shard indices.

## Evaluation

Every metric below is evaluated **on all 165 videos in one combined input**, with VBench `mode="custom_input"`, `local=True`, `read_frame=False`. Imaging quality uses `imaging_quality_preprocessing_mode="longer"`.

| CLI metric | Evaluator model | Videos |
|---|---|---:|
| `imaging_quality` | MUSIQ, through `pyiqa` | 165 |
| `aesthetic_quality` | LAION aesthetic predictor with CLIP ViT-L/14 features | 165 |
| `motion_smoothness` | AMT frame interpolation | 165 |
| `overall_consistency` | ViCLIP video/text alignment | 165 |
| `subject_consistency` | DINO image features | 165 |

For a selected-metric rerun:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task t2v --root runs/t2v-mxfp8 \
  --metrics imaging_quality aesthetic_quality
```

This computes only the requested metrics; it is not a complete five-metric report. First-time evaluation can download evaluator weights. Stage them under the upstream cache conventions if network access is restricted. No evaluator weights are bundled here.

### Observed scores

| Metric | BF16 | MXFP8 | NVFP4 |
|---|---:|---:|---:|
| `imaging_quality` | 0.721889 | 0.722244 | 0.719336 |
| `aesthetic_quality` | 0.622004 | 0.621066 | 0.609699 |
| `motion_smoothness` | 0.980194 | 0.980503 | 0.980903 |
| `overall_consistency` | 0.237381 | 0.238093 | 0.238401 |
| `subject_consistency` | 0.951550 | 0.952117 | 0.947457 |

Rounded from [the compact result artifact](../results/t2v.json), not from means of the 72/93 prompt groups. One sample per prompt, selected prompt groups, and five selected dimensions do not constitute the full standard VBench protocol. Small differences are descriptive, not significance estimates.

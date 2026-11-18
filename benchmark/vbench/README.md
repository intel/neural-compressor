# Wan2.2 quality evaluation with VBench

Generate videos from fixed VBench inputs and evaluate BF16, MXFP8, NVFP4, or
NVFP4 E5M3 Wan2.2-A14B Diffusers checkpoints. The shared driver follows the
`benchmark/evalscope` and `benchmark/lm_eval` `--model_path=...` convention.
Task and precision are read from the checkpoint, and generation, verification,
and evaluation run with one command.

## Setup

Use Linux with NVIDIA GPUs and enough memory for **one complete model per GPU**
(the measured system used B200). FFmpeg/ffprobe and a working CUDA compiler must
be on `PATH`. Inference requires the official vLLM-Omni checkout at
`7465c0f52bcb46a55f62c1b8613846cda30d0083`:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
git -C vllm-omni checkout 7465c0f52bcb46a55f62c1b8613846cda30d0083
export OMNI_SOURCE="$PWD/vllm-omni"
```

Follow [environment setup](docs/environment.md) for the matching vLLM, Torch,
FlashInfer, CUTLASS DSL and evaluator dependencies. The observed environment was
an overlay; a clean installation has not been validated. `requirements-eval.txt`
is a historical inventory, **not an installation requirements file**. An optional
best-effort isolated bootstrap is available:

```bash
bash benchmark/vbench/scripts/setup_env.sh --install .venv-vbench "$OMNI_SOURCE"
source .venv-vbench/bin/activate
```

For **NVFP4 E5M3**, also install the same repository's Omni quantization plugin
in that environment (its entry point loads automatically in Omni workers):

```bash
python -m pip install --no-deps -e benchmark/vllm-qdq-plugin
```

If plugin allowlists are configured, include `inc_nvfp4_e5m3` in the Omni
allowlist. Use a packed AutoRound `nvfp4_v2` checkpoint with
`packing_format=auto_round:llm_compressor_nvfp4_e5m3`, dynamic activations and
16-element groups in both transformers. Native NVFP4 and E5M3 are validated
separately. These scripts evaluate existing exports; they do not quantize or
repair weights automatically. The original E5M3 repair utility is retained at
[scripts/repair_e5m3.py](scripts/repair_e5m3.py) for exports needing the original
BF16 `proj_out` restored; use the exact matching source checkpoint.

## Commands for model cards

Run from the neural-compressor repository root after activating the environment.
A model can be a local directory or a Hub ID; `--revision=FULL_HUB_COMMIT` pins
Hub downloads. Use the same model revision throughout a run.

```bash
# T2V: generate all 165 videos, verify, then evaluate all five metrics.
bash benchmark/vbench/run_vbench.sh --model_path=/path/to/t2v-model

# I2V: download the official cropped references once, then evaluate 355 pairs.
bash benchmark/vbench/scripts/download_i2v.sh ./vbench-data
export I2V_IMAGES="$PWD/vbench-data/crop/16-9"
bash benchmark/vbench/run_vbench.sh --model_path=/path/to/i2v-model
```

The same commands work for all four precisions. To reproduce another precision,
change only `--model_path`. For example:

```bash
bash benchmark/vbench/run_vbench.sh \
  --model_path=intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound \
  --output_dir=runs/t2v-nvfp4 --gpus=0,1,2,3
```

`--gpus` defaults to `CUDA_VISIBLE_DEVICES`, or `0` when unset; use physical GPU
indices. Workers shard inputs deterministically and reuse one engine each.
Evaluation runs on the first selected GPU after all workers exit. This is input
parallelism, with one full model on each GPU. Repeating a command resumes valid
videos and cached metrics. Worker logs and exit records are in `OUTPUT/logs`.

By default the output is `runs/TASK-PRECISION-CHECKPOINT_PATH_HASH`. For a published
model card, specify `--output_dir` explicitly, and retain `configuration.json`,
`task.json`, `jobs.json`, and `evaluation/protocol.json` alongside the scores.
Model weights must remain immutable: provenance includes configuration hashes and
weight sizes, rather than hashes of all weight bytes.

```bash
# Generate now and score later; the default stage is all.
bash benchmark/vbench/run_vbench.sh --model_path=/path/to/model \
  --output_dir=runs/my-model --stage=generate --gpus=0,1

# Evaluate an existing complete run without loading the generation model.
bash benchmark/vbench/run_vbench.sh --stage=evaluate --output_dir=runs/my-model

# Evaluate only selected metrics (not a complete five-metric report).
bash benchmark/vbench/run_vbench.sh --stage=evaluate --output_dir=runs/my-model \
  --metrics=subject_consistency,motion_smoothness
```

Scores are written to `OUTPUT/evaluation/summary.json`, per-video metric details
to `OUTPUT/evaluation/METRIC/result.json`, and the full completion marker to
`OUTPUT/evaluation/completed.json`. Incomplete or inconsistent video sets fail
verification before scoring. Advanced prepare/generate/verify/environment commands
and one-video smoke runs remain available through
`python benchmark/vbench/scripts/benchmark.py --help`.

## Reproduction protocol

These are selected-input, **one video per input** comparisons, not the full
standard VBench protocol or an aggregate VBench score. All precisions use the same
fixed inputs and sampling parameters within each task.

| Setting | T2V | I2V |
| --- | --- | --- |
| Inputs | 165 prompts (72 subject + 93 overall) | 355 pairs (246 subject + 109 background) |
| Resolution | 1280 × 720 | 832 × 480; official 16:9 crops |
| Frames / fps / steps | 81 / 16 / 40 | 81 / 16 / 40 |
| Seed per input | 42 | 0 |
| Guidance scales | 4.0 / 3.0 | 3.5 / 3.5 |
| Boundary ratio / flow shift | 0.875 / 5.0 | 0.9 / 3.0 |
| Execution | TRTLLM_ATTN, eager, no diffusion cache | TRTLLM_ATTN, eager, no diffusion cache |

The official Wan negative prompt is preserved verbatim in the runner. T2V uses
the Omni API with explicit `flashinfer_cutlass`; I2V reuses the unchanged official
example with one cached engine per worker. T2V computes imaging quality, aesthetic
quality, motion smoothness, overall consistency and subject consistency on all
165 videos together. I2V filters each dimension by its official tags: i2v subject,
subject consistency and motion smoothness use 246 videos each; background
consistency and i2v background use 109 each. Camera motion is excluded.

Details: [T2V](docs/t2v.md), [I2V](docs/i2v.md),
[evaluator checkpoint downloads](docs/evaluator-checkpoints.md),
[model IDs and export provenance](docs/quantization.md), and
[data attribution](data/README.md). Older detailed command examples use paths
relative to this `benchmark/vbench` directory.

## Source and verification

Adapted from [changwangss/wan22-quantization-vbench](https://github.com/changwangss/wan22-quantization-vbench)
at `f1527e9755b90456821010bc7e7a98635b05579c`, including the local E5M3 validation
and repair additions present during migration. Historical BF16/MXFP8/NVFP4
results are retained in `results/`; no E5M3 scores are inferred from those results.
See [NOTICE](NOTICE) for attribution. Images, model/evaluator weights and generated
videos are external data and are not bundled.

```bash
python -S -m unittest discover -s benchmark/vbench/tests -v
```

The CPU suite validates protocol, evaluation mapping and resume guards. It does
not establish GPU kernel compatibility or a fresh end-to-end benchmark result.

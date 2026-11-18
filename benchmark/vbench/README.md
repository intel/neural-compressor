# Wan2.2 VBench

Generate and score Wan2.2-A14B Diffusers videos with BF16, MXFP8, NVFP4,
or NVFP4 E5M3 checkpoints. The runner infers the task and precision.

## Setup

Use Linux, NVIDIA GPUs with room for one full model per GPU, and `ffmpeg` and
`ffprobe` on `PATH`. Set `OMNI_SOURCE` to a checkout of
[vLLM-Omni at `7465c0f5`](https://github.com/vllm-project/vllm-omni/tree/7465c0f52bcb46a55f62c1b8613846cda30d0083).
Follow the [environment instructions](https://github.com/changwangss/wan22-quantization-vbench/blob/f1527e9755b90456821010bc7e7a98635b05579c/docs/environment.md)
for the matching inference stack and VBench dependencies. The recorded
environment used B200 GPUs. A fresh install of the recorded stack remains unverified.

For NVFP4 E5M3, install the Omni plugin in the same Python environment
from the neural-compressor repository root.

```bash
python -m pip install --no-deps -e benchmark/vllm-qdq-plugin
```

For I2V, extract the [official cropped image archive](https://drive.google.com/file/d/1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ/view)
and set `I2V_IMAGES` to the extracted `crop/16-9` directory.

## Reproduce

From the neural-compressor repository root, with the environment above active,
pass a local checkpoint directory or Hugging Face model ID.

```bash
bash benchmark/vbench/run_vbench.sh --model_path=org/model
```

Use `--revision=FULL_HUB_COMMIT` to pin a Hub checkpoint. Add `--gpus=0,1,2,3`
to shard generation across GPUs. By default the runner uses the indices in
`CUDA_VISIBLE_DEVICES`, or GPU 0 when unset. The first selected GPU scores the
verified video set. Repeating the command with the same checkpoint and output
directory resumes valid videos and cached scores.

```bash
# Generate first and score later.
bash benchmark/vbench/run_vbench.sh --model_path=org/model --output_dir=runs/model --stage=generate
bash benchmark/vbench/run_vbench.sh --output_dir=runs/model --stage=evaluate
```

Scores are in `<output_dir>/evaluation/summary.json`. The runner prints the
output path at startup and defaults to `runs/TASK-PRECISION-CHECKPOINT_PATH_HASH`.

## Protocol

Each input produces one video with 81 frames, 16 fps, and 40 denoising steps.

| Task | Inputs | Resolution | Evaluation |
| --- | ---: | --- | --- |
| T2V | 165 prompts | 1280 × 720 | Imaging quality, aesthetic quality, motion smoothness, overall consistency, and subject consistency, each on all 165 videos |
| I2V | 355 prompt/image pairs | 832 × 480 | I2V subject, subject consistency, and motion smoothness on 246 subject videos. Background consistency and I2V background on 109 background videos. |

These scores cover five selected dimensions per task. See the original
[benchmark repository](https://github.com/changwangss/wan22-quantization-vbench/tree/f1527e9755b90456821010bc7e7a98635b05579c)
for historical scores, model IDs, sampling settings, and evaluator checkpoint
instructions. The fixed prompt and metadata snapshots are derived from
[Vchitect/VBench](https://github.com/Vchitect/VBench) under its
[Apache-2.0 license](https://github.com/Vchitect/VBench/blob/master/LICENSE).

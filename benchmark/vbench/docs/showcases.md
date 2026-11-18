# Optional qualitative showcases

[Overview](../README.md) · [Environment](environment.md) · [Model IDs and export recipes](quantization.md)

These single-prompt videos are **outside the benchmark sets**. They illustrate
appearance, not VBench scores, speed, memory savings, or general quality recovery.
For the complete measured workflow, use [T2V](t2v.md) or [I2V](i2v.md): prepare
the fixed inputs, generate all videos, verify, then evaluate. See
[validation status](validation.md) for historical full runs versus portable-runner
testing; this showcase wrapper has not been GPU-run as part of this change.

## Run either task

Use the configured [environment](environment.md), a downloaded local checkpoint
from the [six-model table](quantization.md#model-ids), and an unchanged official
[vLLM-Omni checkout at 7465c0f52bcb46a55f62c1b8613846cda30d0083](https://github.com/vllm-project/vllm-omni/tree/7465c0f52bcb46a55f62c1b8613846cda30d0083).
From this repository's root:

```bash
export OMNI_SOURCE=/path/to/vllm-omni
CUDA_VISIBLE_DEVICES=0 bash scripts/showcase.sh t2v \
  /path/to/local/t2v-nvfp4 showcases/t2v-nvfp4/video.mp4 "$OMNI_SOURCE"
CUDA_VISIBLE_DEVICES=0 bash scripts/showcase.sh i2v \
  /path/to/local/i2v-nvfp4 showcases/i2v-nvfp4/video.mp4 "$OMNI_SOURCE"
```

The [wrapper](../scripts/showcase.sh) takes `task local-model output-path omni-source`.
To compare BF16, MXFP8, and NVFP4, rerun with the matching task's checkpoint and
a different output path; no precision flag or checkpoint metadata rewrite is needed.
It checks the source commit, rejects merge conflicts and tracked changes, selects
that checkout through `PYTHONPATH`, and refuses an existing output. Keep generated
media external/untracked and outside benchmark run roots.

For I2V, it downloads the [official white-cat image](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG)
to **input.jpg beside the output video**, using `curl` to a temporary file followed
by a no-overwrite move. A nonempty existing input is reused, not replaced: ensure
it is the same official image, or use a fresh output directory. Model weights,
image terms, and generated-media review remain the user's responsibility.

## Exact settings

Both tasks use the full official Wan negative prompt embedded verbatim in the
wrapper, **81 frames, 40 inference steps, 16 fps, eager execution**, and
`DIFFUSION_ATTENTION_BACKEND=TRTLLM_ATTN`. No diffusion cache, CPU offload, or VAE
tiling/slicing is enabled.

| Setting | T2V: boxing cats | I2V: white cat on a surfboard |
|---|---|---|
| Official entry | [text_to_video.py](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/examples/offline_inference/text_to_video/text_to_video.py) | [image_to_video.py](https://github.com/vllm-project/vllm-omni/blob/7465c0f52bcb46a55f62c1b8613846cda30d0083/examples/offline_inference/image_to_video/image_to_video.py) |
| Width × height | 1280 × 720 | Automatic **544 × 720** for the official image |
| CFG / high-noise CFG | 4.0 / 3.0 | 3.5 / 3.5 |
| Boundary ratio / flow shift | 0.875 / 5.0 | 0.9 / 3.0 |
| Seed | 42 | 0 |
| Startup / stage timeout | Official defaults | 7200 / 7200 seconds |

T2V prompt: “Two anthropomorphic cats in comfy boxing gear and bright gloves fight
intensely on a spotlighted stage.” I2V uses the **full white-cat paragraph** from
the published cards, not a shortened caption. The official I2V entry derives
dimensions with `max_area=480*832`, rounded down to multiples of 16; the benchmark
instead forces **832 × 480** with cropped VBench reference images.

The original T2V **benchmark uses the Omni Python API**, explicitly requesting
`linear_backend="flashinfer_cutlass"`. It does not use this showcase entry.
Inspection of the pinned official T2V parser confirms it exposes **no
`--linear-backend` option**, so this wrapper does not pass one or patch the example;
I2V also receives no forced linear backend. Do not claim identical backend selection
or bitwise reproduction of the historical T2V videos from this CLI alone.

Keep the official output normalization/export code unchanged. It handles tensor
layout and value ranges before MP4 export; directly casting raw Wan floating-point
output to `uint8` can corrupt brightness/colors. This numeric safety is not content
moderation: review generated videos before use or publication.

## Existing published comparisons

These are external model-card assets, not repository files or benchmark inputs.
Each row links the published Hugging Face model ID and its existing videos.

| Task / model card | BF16 reference | Quantized video |
|---|---|---|
| T2V · [intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound) | [Boxing cats](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound/resolve/main/assets/cat_boxing_bf16.mp4) | [MXFP8](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound/resolve/main/assets/cat_boxing_mxfp8.mp4) |
| T2V · [intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound) | [Boxing cats](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound/resolve/main/assets/cat_boxing_bf16.mp4) | [NVFP4](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound/resolve/main/assets/cat_boxing_nvfp4.mp4) |
| I2V · [intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound) | [White cat](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound/resolve/main/assets/bf16.mp4) | [MXFP8](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound/resolve/main/assets/mxfp8.mp4) |
| I2V · [intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound) | [White cat (same reference)](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound/resolve/main/assets/bf16.mp4) | [NVFP4](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound/resolve/main/assets/nvfp4.mp4) |

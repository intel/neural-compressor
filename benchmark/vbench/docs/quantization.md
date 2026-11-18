# Quantization provenance

## Model IDs

| Task | BF16 | MXFP8 | NVFP4 |
|---|---|---|---|
| T2V | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | [intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound) | [intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound](https://huggingface.co/intel-ai/Wan2.2-T2V-A14B-Diffusers-NVFP4-AutoRound) |
| I2V | [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound) | [intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound) |

These IDs are not immutable artifact pins. Follow the base-model and quantized-model upstream licenses; the I2V MXFP8 card records Apache-2.0. No new exports were run for this repository.

## NVFP4: recorded export commands

The historical T2V and I2V wrappers both record [AutoRound commit a85997e19d97762172155c93c4dfc3fcc3976e71](https://github.com/intel/auto-round/tree/a85997e19d97762172155c93c4dfc3fcc3976e71) in their launch logs. Thus `a85997e19` is verified for both tasks, not inferred from the output-directory name. Exported checkpoint metadata records `autoround_version: 0.17.0`, W4A4, weight/activation group size 16, dynamic activation quantization, and packing format `auto_round:llm_compressor`.

The source wrappers were named `run_wan22_nvfp4_rtn_n32_gpu7.sh` and `run_wan22_i2v_nvfp4_rtn_n32_gpu1.sh`. Below, model locations are replaced with the corresponding public IDs, output locations with portable relative directories, and the one visible GPU with GPU 0. The historical physical GPUs were 7 (T2V) and 1 (I2V); both commands used logical `--device_map 0`. Run from the pinned AutoRound checkout with its CLI installed. All quantization options below are retained from the wrappers.

### T2V

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
auto-round \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --scheme NVFP4 \
  --algorithm rtn \
  --disable_opt_rtn \
  --iters 0 \
  --dataset opens2v \
  --nsamples 32 \
  --batch_size 1 \
  --calib_num_inference_steps 8 \
  --num_inference_steps 40 \
  --guidance_scale 5.0 \
  --generator_seed 42 \
  --seed 42 \
  --device_map 0 \
  --low_gpu_mem_usage \
  --diffusion_tuning_cache_size auto \
  --disable_model_free \
  --format auto_round \
  --output_dir exports/wan22-t2v-nvfp4-rtn-n32
```

### I2V

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
auto-round \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --scheme NVFP4 \
  --algorithm rtn \
  --disable_opt_rtn \
  --iters 0 \
  --dataset opens2v \
  --nsamples 32 \
  --batch_size 1 \
  --calib_num_inference_steps 8 \
  --num_inference_steps 40 \
  --guidance_scale 5.0 \
  --generator_seed 42 \
  --seed 42 \
  --device_map 0 \
  --low_gpu_mem_usage \
  --diffusion_tuning_cache_size auto \
  --disable_model_free \
  --format auto_round \
  --output_dir exports/wan22-i2v-nvfp4-rtn-n32
```

These are RTN exports with zero optimization iterations, not AutoRound 200-iteration tuning. The 32 OpenS2V samples and eight calibration inference steps remain part of the recorded recipe even with `--iters 0`. The wrapper explicitly notes that `--diffusion_tuning_cache_size auto` requests SignRound prefetch, but the RTN zero-iteration path does not use that tuning path. Calibration/generation options in these export commands are not the benchmark generation settings: see [T2V](../results/t2v.json) and [I2V](../results/i2v.json) protocols.

The launch commands, revisions, and checkpoint metadata are evidence of the recorded setup; no immutable export-artifact hashes are available here to establish a bitwise chain from each launch to each published checkpoint.

## MXFP8: what is known

- **T2V:** the model owner confirms **AutoRound, 200 iterations**. Do not label it RTN or infer zero-iteration provenance from another recipe. The complete historical export command, calibration details, and exact AutoRound source revision have not been independently reconstructed here; no speculative command is supplied.
- **I2V:** the existing model card records an MXFP8 W8A8 recipe with group size 32 and dynamic activations, quantizing both transformer experts. Its recorded zero-shot RTN recipe is shown below. This is a **recorded card recipe, not independently reconstructed training provenance**: an original export log establishing that this exact command produced the evaluated checkpoint has not been verified. In particular, do not transfer the T2V 200-iteration setting to I2V, or treat the card's `--iters 0` as independently proven artifact history.

```bash
auto-round \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --scheme mxfp8 \
  --iters 0 \
  --disable_opt_rtn \
  --output_dir exports/wan22-i2v-mxfp8-card-recipe
```

The final output directory above is a portable substitute for the card's local output name. The source is the [I2V MXFP8 model card](https://huggingface.co/intel-ai/Wan2.2-I2V-A14B-Diffusers-MXFP8-AutoRound#generate-the-model); card contents can change over time.

## Evaluation is separate from export

The historical inference setup used [vLLM-Omni commit 7465c0f52bcb46a55f62c1b8613846cda30d0083](https://github.com/vllm-project/vllm-omni/tree/7465c0f52bcb46a55f62c1b8613846cda30d0083), VBench 0.1.5, eager execution, TRTLLM attention, and the official Wan negative prompt. AutoRound's export commit is a separate revision, not this inference revision. The compact [observed results](../results/README.md) describe five selected metrics, not an official aggregate VBench score or a full-suite result. No speed or memory claim is inferred from individual showcase videos.

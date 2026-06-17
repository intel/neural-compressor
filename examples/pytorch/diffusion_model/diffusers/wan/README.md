# Step-by-Step

This example offers a unified Wan workflow for quantization and evaluation, covering `t2v` and `i2v` via `main.py`, and `s2v` via `wan_s2v.py`

# Prerequisites

## 1 Environment

You can also use the helper script to install task-specific dependencies:

```bash
# t2v / i2v setup (installs requirements_i2v_t2v.txt and VBench by default)
bash setup.sh --task t2v
bash setup.sh --task i2v

# s2v setup (installs requirements_s2v.txt, skips VBench by default)
bash setup.sh --task s2v
```


## 2 Model Preparation
```bash
# optional: update CLI to latest version
pip install -U "huggingface_hub[cli]"


hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --local-dir /path/to/Wan2.2-T2V-A14B-Diffusers


hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --local-dir /path/to/Wan2.2-I2V-A14B-Diffusers


hf download Wan-AI/Wan2.2-S2V-14B \
  --local-dir /path/to/Wan2.2-S2V-14B
```


## 3 Dataset Preparation

### t2v / i2v 

Both `t2v` and `i2v` use VBench data.
Recommended default is manual preparation for better reproducibility and control.
Use `--vbench_dir=/path/to/VBench` in `run_benchmark.sh` when your VBench repo is not under the default path.

If you prepare VBench manually:

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench
bash vbench2_beta_i2v/download_data.sh
```

Then use:
- t2v:
- `--prompt_folder=/path/to/VBench/prompts/prompts_per_dimension/`
- `--dimension=subject_consistency,overall_consistency`

- i2v:
- `--image_folder=/path/to/VBench/vbench2_beta_i2v/data/crop/16-9`
- `--info_json=/path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json`
- `--dimension=i2v_background,i2v_subject`

### s2v

Recommended default is manual preparation.

```bash
# Step 1: clone EchoMimicV3 repo
git clone https://github.com/antgroup/echomimic_v3.git /path/to/echomimic_v3

# Step 2: build s2v manifest json
python3 prepare_s2v_dataset.py \
  --repo-dir /path/to/echomimic_v3 \
  --manifest-out /path/to/s2v_input_manifest.json

```

The generated /path/to/s2v_input_manifest.json is the s2v input manifest passed via --manifest_path, with image and audio stored as absolute paths.

# Run

##  Quantization

### t2v

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_quant.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-T2V-A14B-Diffusers \
  --task=t2v \
  --output_model=wan_mxfp8_model_t2v
```

### i2v

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_quant.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-I2V-A14B-Diffusers \
  --task=i2v \
  --output_model=wan_mxfp8_model_i2v
```

### s2v

```bash
# Prepare Wan2.2 for s2v
git clone https://github.com/Wan-Video/Wan2.2.git /path/to/Wan2.2
```

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_quant.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-S2V-14B \
  --task=s2v \
  --wan_dir=/path/to/Wan2.2 \
  --output_model=wan_mxfp8_model_s2v
```

Note:
- For `task=s2v`, prepare Wan2.2 manually and pass `--wan_dir=/path/to/Wan2.2` when needed.
- `run_quant.sh` sets `PYTHONPATH` internally for s2v, so you do not need to export it manually.
- For `task=s2v`, `run_quant.sh` dispatches to `wan_s2v.py --quantize` in this example.



## Inference + Evaluation

Note: For `task=t2v/i2v`, prepare VBench manually first, and pass `--vbench_dir=/path/to/VBench` when needed.

### t2v bf16

```bash
bash run_benchmark.sh \
  --topology=wan_bf16 \
  --input_model=/path/to/Wan2.2-T2V-A14B-Diffusers \
  --task=t2v \
  --dimension=subject_consistency,overall_consistency \
  --vbench_dir=/path/to/VBench \
  --output_video_path=wan_t2v_bf16_video \
  --accuracy
```

### t2v mxfp8/fp8

```bash
bash run_benchmark.sh \
  --topology=wan_mxfp8 \
  --input_model=wan_mxfp8_model_t2v \
  --task=t2v \
  --dimension=subject_consistency,overall_consistency \
  --vbench_dir=/path/to/VBench \
  --output_video_path=wan_t2v_mxfp8_video \
  --accuracy
```

### i2v bf16

```bash
bash run_benchmark.sh \
  --topology=wan_bf16 \
  --input_model=/path/to/Wan2.2-I2V-A14B-Diffusers \
  --task=i2v \
  --dimension=i2v_background,i2v_subject \
  --vbench_dir=/path/to/VBench \
  --output_video_path=wan_i2v_bf16_video \
  --accuracy
```

### i2v mxfp8/fp8

```bash
bash run_benchmark.sh \
  --topology=wan_mxfp8 \
  --input_model=wan_mxfp8_model_i2v \
  --task=i2v \
  --dimension=i2v_background,i2v_subject \
  --vbench_dir=/path/to/VBench \
  --output_video_path=wan_i2v_mxfp8_video \
  --accuracy
```

### s2v bf16

Note: For `task=s2v`, prepare Wan2.2 manually and pass `--wan_dir=/path/to/Wan2.2` when needed.

```bash
bash run_benchmark.sh \
  --topology=wan_bf16 \
  --task=s2v \
  --input_model=/path/to/Wan2.2-S2V-14B \
  --wan_dir=/path/to/Wan2.2 \
  --manifest_path=/path/to/s2v_input_manifest.json \
  --output_video_path=wan_s2v_bf16_video \
  --accuracy
```

### s2v mxfp8/fp8

```bash
bash run_benchmark.sh \
  --topology=wan_mxfp8 \
  --task=s2v \
  --input_model=/path/to/Wan2.2-S2V-14B \
  --quantized_model=wan_mxfp8_model_s2v \
  --wan_dir=/path/to/Wan2.2 \
  --manifest_path=/path/to/s2v_input_manifest.json \
  --output_video_path=wan_s2v_mxfp8_video \
  --accuracy
```

When `task=s2v` and `--accuracy` is set, `run_benchmark.sh` will run generation via `wan_s2v.py`, then run `evaluate_manifest_no_gt.py`.

- Optional eval arg: `--s2v_eval_output` (default: `${output_video_path}/evaluation_no_gt_metrics_s2v.json`)
- Internal defaults: matched manifest `${output_video_path}/s2v_manifest_with_generate_video.json`, `max_frames=32`, `metric_size=192`


Metric note: current s2v benchmark manifest does not provide ground-truth videos, so `FID` and `FVD` are not computed. The script reports proxy metrics from available image/audio/generated-video signals (for example SSIM, PSNR, Sync-C, HKC, HKV, CSIM, EFID).

For sharding and multi-GPU execution, set `--gpu_ids` (for example `--gpu_ids=0,1,2,3`) or set `CUDA_VISIBLE_DEVICES` before running `run_benchmark.sh`.


### Standalone Accuracy Evaluation (Optional)

If you already use `--accuracy` in `run_benchmark.sh`, you can skip this section.
Use this section when you want to evaluate existing videos without re-running generation.

```bash
# t2v accuracy on generated videos
cd /path/to/VBench
python evaluate.py \
  --dimension subject_consistency motion_smoothness aesthetic_quality overall_consistency imaging_quality \
  --videos_path /path/to/wan_t2v_bf16_video \
  --mode vbench_standard

# i2v accuracy on generated videos
cd /path/to/VBench
python evaluate_i2v.py \
  --dimension i2v_background i2v_subject subject_consistency background_consistency motion_smoothness \
  --videos_path /path/to/wan_i2v_bf16_video \
  --ratio 16-9 \
  --mode vbench_standard

# s2v standalone evaluation from generated manifest.
python evaluate_manifest_no_gt.py \
  --manifest /path/to/wan_s2v_output/s2v_manifest_with_generate_video.json \
  --output /path/to/wan_s2v_output/evaluation_no_gt_metrics_s2v.json \
  --max_frames 32 \
  --metric_size 192
```


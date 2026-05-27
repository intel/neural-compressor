# Step-by-Step

This example provides a unified Wan entry for quantization and evaluation, with both t2v and i2v support.

# Prerequisite

## 1. Environment

```shell
pip install -r requirements.txt
# Use latest dev branch if needed before release
# INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@master
# pip install git+https://github.com/intel/auto-round.git@main
pip install neural-compressor-pt==3.7
pip install auto-round
```

## 2. Prepare Model

Use a local Wan diffusers model path, for example:

- Wan2.2-T2V-A14B-Diffusers
- Wan2.2-I2V-A14B-Diffusers

Download example (from Hugging Face):

```bash
pip install -U "huggingface_hub[cli]"

# t2v model
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --local-dir /path/to/Wan2.2-T2V-A14B-Diffusers

# i2v model
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --local-dir /path/to/Wan2.2-I2V-A14B-Diffusers
```

## 3. Prepare Dataset

Clone VBench and prepare the required data:

```bash
git clone https://github.com/Vchitect/VBench.git
```

- t2v: pass prompt folder with --prompt_folder, and set --dimension to select `${prompt_folder}/${dimension}.txt`
- t2v: can pass --dimension for evaluation filtering (validated dimensions include `subject_consistency` and `overall_consistency`)
- i2v: pass --image_folder, --info_json, and --dimension (validated dimensions include `i2v_subject`, `i2v_background`, `subject_consistency`, `background_consistency`, and `motion_smoothness`)

# Run

## Quantization

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_quant.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-T2V-A14B-Diffusers \
  --task=t2v \
  --output_model=wan_mxfp8_model
```

## Inference + Evaluation

### t2v BF16

```bash
bash run_benchmark.sh \
  --topology=wan_bf16 \
  --input_model=/path/to/Wan2.2-T2V-A14B-Diffusers \
  --task=t2v \
  --dimension=subject_consistency \
  --prompt_folder=/path/to/VBench/prompts/prompts_per_dimension/ \
  --output_video_path=wan_bf16_video
```

### t2v FP8 / MXFP8

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_benchmark.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-T2V-A14B-Diffusers \
  --quantized_model=wan_mxfp8_model \
  --task=t2v \
  --dimension=overall_consistency \
  --prompt_folder=/path/to/VBench/prompts/prompts_per_dimension/ \
  --output_video_path=wan_mxfp8_video
```

### i2v BF16

For i2v BF16, provide --image_folder, --info_json, and --dimension explicitly:

```bash
bash run_benchmark.sh \
  --topology=wan_bf16 \
  --input_model=/path/to/Wan2.2-I2V-A14B-Diffusers \
  --task=i2v \
  --dimension=i2v_subject \
  --image_folder=/path/to/VBench/vbench2_beta_i2v/data/crop/16-9 \
  --info_json=/path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --output_video_path=wan_i2v_bf16_video 
```

### i2v FP8 / MXFP8

```bash
# topology supports wan_mxfp8 or wan_fp8
bash run_benchmark.sh \
  --topology=wan_mxfp8 \
  --input_model=/path/to/Wan2.2-I2V-A14B-Diffusers \
  --quantized_model=wan_mxfp8_model \
  --task=i2v \
  --dimension=i2v_background \
  --image_folder=/path/to/VBench/vbench2_beta_i2v/data/crop/16-9 \
  --info_json=/path/to/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --output_video_path=wan_i2v_mxfp8_video
```

## Accuracy Evaluation

For evaluating existing videos, call VBench directly.

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
  --mode vbench_standard
```

# Notes

- Quantized weights are saved under:
  - <output_model>/transformer
  - <output_model>/transformer_2


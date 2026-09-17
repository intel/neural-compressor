# Stable Diffusion XL MXFP8

This example quantizes the UNet in
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
to MXFP8 with AutoRound and evaluates generated images on COCO2014 captions.

MXFP8 uses a block size of 32 and E8M0 power-of-two scales. The exported directory is a complete Diffusers
pipeline: the UNet weights are fake-quantized and UNet linear-layer activations are dynamically quantized during
inference.

## Prerequisites

Install a CUDA-compatible PyTorch build first, then install the example dependencies, Neural Compressor, and
AutoRound:

```bash
pip install -r requirements.txt
pip install neural-compressor-pt
pip install git+https://github.com/intel/auto-round.git@main
```

Download the COCO2014 caption file used by MLPerf Stable Diffusion evaluation:

```bash
wget https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv
```

Access to the gated SDXL model on Hugging Face may require authentication.

## Quantization

Run calibration-free pure RTN quantization:

```bash
bash run_quant.sh \
    --input_model=stabilityai/stable-diffusion-xl-base-1.0 \
    --output_model=./saved_results
```

The recipe fixes `iters=0` and `disable_opt_rtn=True`, so it does not require calibration samples or tuning.
The equivalent Python command is:

```bash
python main.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --output_dir ./saved_results \
    --scheme MXFP8 \
    --disable_opt_rtn \
    --quantize
```

## Evaluation

Generate the BF16 baseline on all visible GPUs and calculate CLIP, CLIP-IQA, and ImageReward:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh \
    --input_model=stabilityai/stable-diffusion-xl-base-1.0 \
    --dataset_location=captions_source.tsv \
    --output_image_path=./tmp_imgs_bf16
```

Evaluate the MXFP8 pipeline:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh \
    --input_model=stabilityai/stable-diffusion-xl-base-1.0 \
    --quantized_model=./saved_results \
    --dataset_location=captions_source.tsv \
    --output_image_path=./tmp_imgs_mxfp8
```

When multiple GPUs are listed, the script partitions the TSV into balanced subsets. Every sample is assigned
exactly once, even when the sample count is not divisible by the GPU count. Use `--limit=10` for a short smoke
test.

## Results

The following results were measured on 5,000 COCO2014 validation prompts. Higher is better for every metric.

| Format | CLIP ↑ | CLIP-IQA ↑ | ImageReward ↑ |
|---|---:|---:|---:|
| BF16 | 26.8583 | 0.929027 | 0.812544 |
| MXFP8 (RTN) | 26.8491 | 0.930057 | 0.814659 |

Compared with BF16, MXFP8 retains an average of 100.11% across the three metrics.

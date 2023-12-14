# Step-by-Step

This is an experimental example to quantize unet model. We use dummy data to do quantization and evaluation, so the accuracy is not guaranteed.

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```bash
python prepare_model.py --input_model='CompVis/stable-diffusion-v1-4' --output_model='unet-export.onnx'
```

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=path/to/model \  # model path as *.onnx
                   --output_model=path/to/save
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --batch_size=batch_size \
                      --mode=performance
```

# Step-by-Step (Deprecated)

This example load an image classification model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on MNIST dataset.

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --output_model='mnist-12.onnx'
```

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=path/to/model \  # model path as *.onnx
                   --dataset_location=/path/to/mnist \ # if dataset doesn't exist, it will be downloaded automatically into this path
                   --output_model=path/to/save
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/mnist \ # if dataset doesn't exist, it will be downloaded automatically into this path
                      --mode=performance # or accuracy
```

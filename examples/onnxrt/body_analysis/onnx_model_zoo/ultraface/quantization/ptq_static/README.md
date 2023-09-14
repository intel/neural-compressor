# Step-by-Step

This example load an model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [WIDER FACE dataset (Validation Images)](http://shuoyang1213.me/WIDERFACE/).

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --input_model='version-RFB-640.onnx' --output_model='version-RFB-640-12.onnx'
```

## 3. Prepare Dataset

Download dataset [WIDER FACE dataset (Validation Images)](http://shuoyang1213.me/WIDERFACE/).

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --dataset_location=/path/to/data \
                   --output_model=path/to/save
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/data \
                      --mode=performance # or accuracy
```

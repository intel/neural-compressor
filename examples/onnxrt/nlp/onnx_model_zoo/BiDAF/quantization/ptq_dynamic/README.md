Step-by-Step
============

This example loads a neural network for answering a query about a given context paragraph. It is converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/).

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Download model from [ONNX Model Zoo](https://github.com/onnx/models).

```shell
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx
```

Convert opset version to 11 for more quantization capability.
```python
import onnx
from onnx import version_converter

model = onnx.load('bidaf-9.onnx')
model = version_converter.convert_version(model, 11)
onnx.save_model(model, 'bidaf-11.onnx')
```

## 3. Prepare Dataset
Download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).

Dataset directories:

```bash
squad
├── dev-v1.1.json
└── train-v1.1.json
```

# Run

## 1. Quantization

Dynamic quantization:

```bash
bash run_quant.sh --input_model=path/to/model \ # model path as *.onnx
                   --dataset_location=path/to/squad/dev-v1.1.json
                   --output_model=path/to/model_tune
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # model path as *.onnx
                      --dataset_location=path/to/squad/dev-v1.1.json
                      --mode=performance # or accuracy
```

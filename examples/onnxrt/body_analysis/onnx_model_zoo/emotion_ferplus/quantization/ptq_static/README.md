Step-by-Step
============

This example load an model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [Emotion FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

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
wget https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
```

Convert opset version to 12 for more quantization capability.

```python
import onnx
from onnx import version_converter
model = onnx.load('emotion-ferplus-8.onnx')
model = version_converter.convert_version(model, 12)
onnx.save_model(model, 'emotion-ferplus-12.onnx')
```

## 3. Prepare Dataset
Download dataset [Emotion FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

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

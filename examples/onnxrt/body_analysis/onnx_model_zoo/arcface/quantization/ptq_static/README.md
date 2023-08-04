Step-by-Step
============

This example load a face recognition model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [Refined MS-Celeb-1M](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip).

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
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx
```

Convert opset version to 11 for more quantization capability.

```python
import onnx
from onnx import version_converter
model = onnx.load('arcfaceresnet100-8.onnx')
model = version_converter.convert_version(model, 11)
onnx.save_model(model, 'arcfaceresnet100-11.onnx')
```

## 3. Prepare Dataset
Download dataset [Refined MS-Celeb-1M](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip).

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=path/to/model \  # model path as *.onnx
                   --dataset_location=/path/to/faces_ms1m_112x112/task.bin \
                   --output_model=path/to/save
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/faces_ms1m_112x112/task.bin \
                      --mode=performance # or accuracy
```

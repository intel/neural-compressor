# Step-by-Step

This example load a face recognition model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [Refined MS-Celeb-1M](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip).

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --input_model='arcfaceresnet100-8.onnx' --output_model='arcfaceresnet100-11.onnx'
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

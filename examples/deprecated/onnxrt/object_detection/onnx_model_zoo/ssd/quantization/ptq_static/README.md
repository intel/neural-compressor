# Step-by-Step (Deprecated)

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download).

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --output_model='ssd-12.onnx'
```

## 3. Prepare Dataset

Download [MS COCO 2017 dataset](https://cocodataset.org/#download).

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \ # dataset path containing 'val2017' and 'annotations' folders
                   --label_path=label_map.yaml \
                   --quant_format="QOperator"
```

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \ # dataset path containing 'val2017' and 'annotations' folders
                   --label_path=label_map.yaml \
                   --quant_format="QDQ"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=path/to/val2017 \ # dataset path containing 'val2017' and 'annotations' folders
                      --label_path=label_map.yaml \
                      --mode=performance # or accuracy
```

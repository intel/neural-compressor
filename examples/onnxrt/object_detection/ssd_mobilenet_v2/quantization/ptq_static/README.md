# Step-by-Step

This example load an object detection model converted from Tensorflow and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download).

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```shell
python prepare_model.py --output_model='ssd_mobilenet_v2_coco_2018_03_29.onnx'
```

> Note: For now, use "onnx==1.14.1" in this step in case you get an error `ValueError: Could not infer attribute explicit_paddings type from empty iterator`. Refer to this [link](https://github.com/onnx/tensorflow-onnx/issues/2262) for more details of this error.

## 3. Prepare Dataset

Download [MS COCO 2017 dataset](https://cocodataset.org/#download).

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \
                   --quant_format="QOperator"
```

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \
                   --quant_format="QDQ"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=path/to/val2017 \
                      --mode=performance # or accuracy
```

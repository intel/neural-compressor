Step-by-Step
============

This example show how to export, quantize and evaluate 2 [DETR](https://huggingface.co/docs/transformers/model_doc/detr) R18 models on [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) dataset, one for table detection and one for table structure recognition, dubbed Table Transformers.

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
bash prepare.sh
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

Download PubTables-1M dataset according to this [doc](https://github.com/microsoft/table-transformer/tree/main#training-and-evaluation-data).
After downloading and extracting, PubTables-1M dataset folder should contain `PubTables-1M-Structure` and `PubTables-1M-Detection` folders.

## 3. Prepare Model

Prepare DETR R18 model for table structure recognition.

```
python prepare_model.py  --input_model=structure_detr  --output_model=pubtables1m_structure_detr_r18.onnx --dataset_location=/path/to/pubtables-1m
```

Prepare DETR R18 model for table detection.
```
python prepare_model.py  --input_model=detection_detr  --output_model=pubtables1m_detection_detr_r18.onnx --dataset_location=/path/to/pubtables-1m
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=/path/to/pubtables-1m # dataset_folder should contains `PubTables-1M-Structure` and/or `PubTables-1M-Detection` folders
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=/path/to/pubtables-1m # dataset_folder should contains `PubTables-1M-Structure` and/or `PubTables-1M-Detection` folders
                      --mode=performance # or accuracy
```

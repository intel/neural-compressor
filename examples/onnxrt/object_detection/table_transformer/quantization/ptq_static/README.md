Step-by-Step
============

This example show how to export, quantize and evaluate the DETR R18 model for table structure recognition task based on PubTables-1M dataset.

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

Download dataset according to this [doc](https://github.com/microsoft/table-transformer/tree/main#training-and-evaluation-data).

## 3. Prepare Model

```shell
wget https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_structure_detr_r18.pth

bash export.sh --input_model=/path/to/pubtables1m_structure_detr_r18.pth \
               --output_model=/path/to/export \ # model path as *.onnx
               --dataset_location=/path/to/dataset_folder # dataset_folder should contains PubTables1M-Structure-PASCAL-VOC and PubTables1M-Table-Words-JSON sub-folders
```

# Run

## Diagnosis
Neural Compressor offers quantization and benchmark diagnosis. Adding `diagnosis` parameter to Quantization/Benchmark config will provide additional details useful in diagnostics.
### Quantization diagnosis
```
config = PostTrainingQuantConfig(
    diagnosis=True,
    ...
)
``` 

### Benchmark diagnosis
```
config = BenchmarkConfig(
    diagnosis=True,
    ...
)
``` 

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=/path/to/dataset_folder # dataset_folder should contains PubTables1M-Structure-PASCAL-VOC and PubTables1M-Table-Words-JSON sub-folders
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=/path/to/dataset_folder # dataset_folder should contains PubTables1M-Structure-PASCAL-VOC and PubTables1M-Table-Words-JSON sub-folders
                      --mode=performance # or accuracy
```

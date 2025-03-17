Step-by-Step
============

This example load a BERT model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/). 

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

download the GLUE data with `prepare_data.sh` script.
```shell
export GLUE_DIR=path/to/glue_data
export TASK_NAME=MRPC

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

## 3. Prepare Model

```shell
python prepare_model.py --input_model='MRPC.zip' --output_model='bert.onnx'
```

# Run

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune \
                   --dataset_location=path/to/glue_data \
                   --quant_format="QOperator"
```

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=path/to/model \ # model path as *.onnx
                   --output_model=path/to/model_tune \ # model path as *.onnx
                   --dataset_location=path/to/glue_data \
                   --quant_format="QDQ"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \ # model path as *.onnx
                      --dataset_location=path/to/glue_data \
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```

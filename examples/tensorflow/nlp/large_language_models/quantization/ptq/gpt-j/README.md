Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor smooth quantization of language models gpt-j-6B.

## Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
pip install -r requirements
```

## 2. Prepare Dataset and Model
The dataset and model will be loaded when applying quantization.

## Run

### Smooth Quantization


```shell
bash run_quant.sh --fp32_path=/path/to/save/fp32/model --output_model=/path/to/save/int8/model
```
The fp32 gpt-j-6B model will be loaded to the ```fp32_path``` as is set.

## Benchmark

### Evaluate performance

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=benchmark
```

### Evaluate accuracy

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=benchmark
```


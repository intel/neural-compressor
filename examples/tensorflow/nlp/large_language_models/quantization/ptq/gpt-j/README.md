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

## 2. Prepare Pretrained model
Run the follow script to download gpt-j-6B saved_model to ```./gpt-j-6B```: 
 ```
bash prepare_model.sh
 ```

## 3. Prepare Dataset
The dataset will be automatically loaded.

## Run

### Smooth Quantization

```shell
bash run_quant.sh --input_model=<FP32_MODEL_PATH> --output_model=<INT8_MODEL_PATH>
```

### Benchmark

#### Evaluate Performance

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=benchmark
```

#### Evaluate Accuracy

```shell
bash run_benchmark.sh --input_model=<MODEL_PATH> --mode=accuracy
```


Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor QAT feature.


# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```
### Install Requirements
The Tensorflow and intel-extension-for-tensorflow is mandatory to be installed to run this QAT example.
The Intel Extension for Tensorflow for Intel CPUs is installed as default.
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

# Run

The baseline model will be generated and pretrained on CIFAR10 dataset. Then, it will be saved to "./baseline_model". The CIFAR10 dataset will be automatically loaded.
To apply QAT, run the command below:

## 1. Quantization
```shell
bash run_tuning.sh --output_model=/path/to/output_model
```

## 2. Benchmark

### Performance
```shell
bash run_benchmark.sh --input_model=/path/to/input_model --mode=performance
```
### Accuracy
```shell
bash run_benchmark.sh --input_model=/path/to/input_model --mode=accuracy
```
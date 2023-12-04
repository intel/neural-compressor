Step-by-Step
============

This document is used to list steps of reproducing Intel® Neural Compressor QAT feature on intel CPU.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

# Prerequisite

## 1. Environment

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### Installation Dependency packages
```shell
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow
Intel Extension for Tensorflow is mandatory to be installed to run this QAT example.
```shell
pip install intel-extension-for-tensorflow[cpu]
```
> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

# Run

The baseline model will be generated and pretrained on CIFAR10 dataset. Then, it will be saved to "./baseline_model". The CIFAR10 dataset will be automatically loaded.
To apply QAT, run the command below:

## 1. Quantization
```shell
bash run_quant.sh --output_model=/path/to/output_model
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
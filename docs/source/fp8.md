Emulated FP8 Quantization
=======
1. [Introduction](#introduction)   
2. [Supported Framework](#supported-framwork)   
3. [Examples](#examples)   
    3.1. [Old API Configuration](#old-api-configuration-for-intel-neural-compressor-1x)   
    3.2. [New API Configuration](#new-api-configuration-for-intel-neural-compressor-20)  
    3.2. [Automatic Tuning Strategy](#automatic-tuning-strategy)  
    3.2. [Global Environment Variables](#global-environment-variables)  

## Introduction
Float point 8(FP8) is a promising data type for low precision quantization. In Intel Neural Compressor, An emulated FP8 quantization is supported in branch [fp8_adaptor](https://github.com/intel/neural-compressor/tree/fp8_adaptor). With specifing precision(fp8_e5m2, fp8_e4m3, fp8_e3m4), users can validate the accuracy of quantized fp8 model.


## Supported Framework

| Framework  | Emulated FP8 Quantization |
|------------|:-------------------------:|
| PyTorch    |          &#10004;         |
| ONNX       |             WIP           |

## Examples

Comparing with int8 quantization, only one parameter: precision(fp8_e5m2/fp8_e4m3/fp8_e3m4) is added.

BTW, for models with BatchNorm, it is recommanded to calibrate its statistics in train mode with fp8 data type before quantization.

### Old API Configuration for Intel Neural Compressor 1.x

```yaml
model:
    name: xxx
    framework: pytorch

quantization:
    approach: post_training_static_quant    # no need for fp8_e5m2
    precision: fp8_e4m3    # allowed precision is fp8_e5m2, fp8_e4m3, fp8_e3m4
    calibration:
        batchnorm_sampling_size: 3000    # only needed for models w/ BatchNorm
        sampling_size: 300

tuning:
    accuracy_criterion:
        relative:  0.01
    exit_strategy:
        timeout: 0
    random_seed: 9527
```

### New API Configuration for Intel Neural Compressor 2.0
```python
quant_conf = PostTrainingQuantConfig(
    precision="fp8_e5m2",
    calibration_sampling_size=[300],
    batchnorm_calibration_sampling_size=[3000],
)
```

### Automatic Tuning Strategy
Unlike the int8 base strategy, the FP8 auto tuning strategy will attempt per operation type tuning. We first aggressively quantize all op types, and if the accuracy requirement is missed, the strategy tries to quantize one op type and accumulates them together. Finally, the user will get the following information.

```log
[INFO] Suggested op types with KL algorithm are: ['Matmul', 'LayerNorm', 'Linear']
[INFO] Suggested FP8 op types are: ['Matmul', 'Embedding', 'LayerNorm', 'Linear']; Accuracy is 0.5560059529291749
```

### Global Environment Variables
In order to facilitate customer customization , some global environment variables are used.

| Framework  | Usage | Supported Values |
|------------|:-------:|:-------------------------:|
| FP8_OP_TYPE_LIST | To specify module type range of emulated FP8 quantization | 'linear', 'conv2d', 'bmm', 'amm', 'mm','add', 'mul', 'div', 'embedding', 'embeddingbag', 'layernorm' |
| DISABLE_FIRST_CONV | Whether quantize the first convolution layer | True/False |
| DISABLE_LAST_LINEAR | Whether quantize the last linear layer | True/False |
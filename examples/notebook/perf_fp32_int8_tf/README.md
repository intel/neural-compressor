# Performance of FP32 Vs. INT8 ResNet50 Model

## Introduction

Intel® Neural Compressor helps user to quantize FP32 model to accelerate the inference. The increase is obviously if running on Xeon with Intel® Deep Learning Boost.

This is one command example supports user test the performance improvement of a quantized ResNet50 model based on Tensorflow by Intel® Neural Compressor, without any code work and prepare work on local server or cloud.

## Steps
1. Download the FP32 and INT8 model of ResNet50 based on Tensorflow.
2. Test the performance (throughput and latency) of both models in same code.
3. Compare the performance and output result in screen print and PNG file.

It uses the dummy dataset to test performance, so no accuracy data is tested. If you want to know the accuracy impact, please refer to other examples.

## Check CPU Support Intel® Deep Learning Boost

To get obviously increase, it's recommended to test quantized model on CPU support Intel® Deep Learning Boost.

Run following command to check in Ubuntu:
```
lscpu | grep avx512_vnni

...
avx512_vnni
...
```

If there is avx512_vnni, that means the CPU supports Intel® Deep Learning Boost.

## Setup Environment
Following commands don't need to run manually. They will be called during running sample automatically.

### Script
```
./set_env.sh

```

### Activate Running Environment
```
source env_intel_tf/bin/activate
```


## Run Sample
```
./run.sh
```

This script will execute following steps:
1. Setup running environment
2. Activate running environment
3. Test performance
4. Compare the result.

## Check Result

1. Screen Print

We will see the result in screen. 

For example:
```
Compare the Performance of FP32 and INT8 Models
Model           FP32                    INT8                    
throughput(fps) 378.35371907536023      X113.26080122625        
latency(ms)     38.190600580098675      Y7.58170614437181       
qt.qpa.xcb: XKeyboard extension not present on the X server

Save to fp32_int8_absolute.png

Model           FP32                    INT8                    
throughput_times1                       X.942381018341494       
latency_times   1                       Y.46036736467385464     

Save to fp32_int8_times.png

```

2. Image File

The result will be shown as figures in PNG files:
Please check them:

  fp32_int8_absolute.png

  fp32_int8_times.png

### Note
1. Code is not optimize for performance.

This sample uses common code to test the performance of FP32 and INT8 ResNet50 models. The code is not designed to optimize to release hardware performance.

To get the better benchmark result, please refer [Maximize TensorFlow* Performance on CPUs](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) 

2. No accuracy data.

This sample doesn't test the accuracy and compare, because it uses dummy dataset.

If you want to know the accuracy lost, please refer to other tutorial of Intel® Neural Compressor.
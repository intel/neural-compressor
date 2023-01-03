Quantization Support Matrix
===========================

This document provides a quantization support matrix for the following frameworks listed below:

| Framework | Backend Library |  Symmetric Quantization | Asymmetric Quantization |
| :-------------- |:---------------:| ---------------:|---------------:|
| TensorFlow    | [oneDNN](https://github.com/oneapi-src/oneDNN) | Activation (int8/uint8), Weight (int8) | - |
| PyTorch         | [FBGEMM](https://github.com/pytorch/FBGEMM) | Activation (uint8), Weight (int8) | Activation (uint8) |
| PyTorch IPEX | [oneDNN](https://github.com/oneapi-src/oneDNN)  | Activation (int8/uint8), Weight (int8) | - |
| MXNet           | [oneDNN](https://github.com/oneapi-src/oneDNN)  | Activation (int8/uint8), Weight (int8) | - |
| ONNX Runtime | [MLAS](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/mlas) | Weight (int8) | Activation (uint8) |

### TensorFlow
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))

### PyTorch
+ Symmetric Quantization
    + int8: scale = max(abs(rmin), abs(rmax)) / (float(max(int8) - min(int8)) / 2)
    + uint8: scale = max(abs(rmin), abs(rmax)) / (float(max(int8) - min(int8)) / 2)
+ Asymmetric Quantization
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = min(uint8)  - round(rmin / scale)

### PyTorch IPEX
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))

### MXNet
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))
	
### ONNX Runtime
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
+ Asymmetric Quantization
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = min(uint8)  - round(rmin / scale)
	
### Reference
+ oneDNN: [Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html)
+ FBGEMM: [FBGEMM Quantization](https://github.com/pytorch/pytorch/blob/master/torch/quantization/observer.py)
+ MLAS:  [MLAS Quantization](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/onnx_quantizer.py)

BigDL Nano Support
===========================

Neural Coder collaborates with [BigDL-Nano](https://bigdl.readthedocs.io/en/latest/doc/Nano/Overview/nano.html), a Python library that automatically applies modern CPU optimizations, to further democratize ease-of-use BigDL-Nano APIs as a **no-code** solution for PyTorch Deep Learning programmers.

## Example
For instance, to perform BF16 + Channels Last optimizations with BigDL-Nano API using Neural Coder on the [example code](../examples/nano/resnet18.py) and run this code with the enabled optimizations, users can simply execute this command:
```
python -m neural_coder -o nano_bf16_channels_last ../examples/nano/resnet18.py
```
The alias for each optimization set is documented in the below Support Matrix. Note that you need to ```pip install bigdl``` first following [BigDL-Nano documentation](https://github.com/intel-analytics/BigDL#installing).

## Support Matrix

| Optimization Set | API Alias | 
| ------------- | ------------- | 
| BF16 + Channels Last | `nano_bf16_channels_last` | 
| BF16 + IPEX + Channels Last | `nano_bf16_ipex_channels_last` | 
| BF16 + IPEX | `nano_bf16_ipex` | 
| BF16 | `nano_bf16` | 
| Channels Last | `nano_fp32_channels_last` | 
| IPEX + Channels Last | `nano_fp32_ipex_channels_last` | 
| IPEX | `nano_fp32_ipex` | 
| Convert CUDA TO GPU | `nano_gpu_to_cpu` | 
| INT8 | `nano_int8` | 
| JIT + BF16 + Channels Last | `nano_jit_bf16_channels_last` | 
| JIT + BF16 + IPEX + Channels Last | `nano_jit_bf16_ipex_channels_last` | 
| JIT + BF16 + IPEX | `nano_jit_bf16_ipex` | 
| JIT + BF16 | `nano_jit_bf16` | 
| JIT + Channels Last | `nano_jit_fp32_channels_last` | 
| JIT + IPEX + Channels Last | `nano_jit_fp32_ipex_channels_last` | 
| JIT + IPEX | `nano_jit_fp32_ipex` | 
| JIT | `nano_jit_fp32` | 
| ONNX Runtime | `nano_onnxruntime_fp32` | 
| ONNX Runtime + INT8 | `nano_onnxruntime_int8_qlinear` | 
| OpenVINO | `nano_openvino_fp32` | 
| OpenVINO + INT8 | `nano_openvino_int8` |

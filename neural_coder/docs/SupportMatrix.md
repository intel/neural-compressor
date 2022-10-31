Supported Optimization Features
===========================

| Framework | Optimization | API Alias |
| ------------- | ------------- | ------------- |
| PyTorch | [Mixed Precision](https://pytorch.org/docs/stable/amp.html) | `pytorch_amp` |
| PyTorch | [Channels Last](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) | `pytorch_channels_last` |
| PyTorch | [JIT (Just-In-Time) Script/Trace](https://pytorch.org/docs/stable/jit.html) & [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html) | `pytorch_jit_script`, `pytorch_jit_trace`, `pytorch_jit_script_ofi`, `pytorch_jit_trace_ofi` |
| PyTorch | JIT with [TorchDynamo](https://github.com/pytorch/torchdynamo) | `pytorch_torchdynamo_jit_script`, `pytorch_torchdynamo_jit_trace`, `pytorch_torchdynamo_jit_script_ofi`, `pytorch_torchdynamo_jit_trace_ofi` |
| PyTorch | [Intel Neural Compressor Mixed Precision](https://github.com/intel/neural-compressor/blob/master/docs/mixed_precision.md) | `pytorch_inc_bf16` | 
| PyTorch | [Intel Neural Compressor INT8 Static Quantization (FX/IPEX)](https://github.com/intel/neural-compressor/blob/master/docs/PTQ.md) | `pytorch_inc_static_quant_fx`, `pytorch_inc_static_quant_ipex` |
| PyTorch | [Intel Neural Compressor INT8 Dynamic Quantization](https://github.com/intel/neural-compressor/blob/master/docs/dynamic_quantization.md) | `pytorch_inc_dynamic_quant` |
| PyTorch | [Intel Extension for PyTorch (FP32, BF16, INT8 Static/Dynamic Quantization)](https://github.com/intel/intel-extension-for-pytorch) | `pytorch_ipex_fp32`, `pytorch_ipex_bf16`, `pytorch_ipex_int8_static_quant`, `pytorch_ipex_int8_dynamic_quant` |
| PyTorch | [Alibaba Blade-DISC](https://github.com/alibaba/BladeDISC) | `pytorch_aliblade` |
| PyTorch Lightning | [Mixed Precision](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html) | `pytorch_lightning_bf16_cpu` |
| TensorFlow | [Mixed Precision](https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html) | `tensorflow_amp` |
| Keras | [Mixed Precision](https://www.tensorflow.org/guide/mixed_precision) | `keras_amp` |
| ONNX Runtime | [INC Static Quantization (QLinear)](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/README.md#operator-oriented-with-qlinearops) | `onnx_inc_static_quant_qlinear` |

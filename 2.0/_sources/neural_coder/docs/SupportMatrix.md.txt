Supported Optimization Features
===========================

| Category | Optimization | API Alias |
| ------------- | ------------- | ------------- |
| PyTorch | [Mixed Precision](https://pytorch.org/docs/stable/amp.html) | `pytorch_amp` |
| PyTorch | [Channels Last](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) | `pytorch_channels_last` |
| PyTorch | [JIT (Just-In-Time) Script/Trace](https://pytorch.org/docs/stable/jit.html) & [optimize_for_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html) | `pytorch_jit_script`, `pytorch_jit_trace`, `pytorch_jit_script_ofi`, `pytorch_jit_trace_ofi` |
| PyTorch | JIT with [TorchDynamo](https://github.com/pytorch/torchdynamo) | `pytorch_torchdynamo_jit_script`, `pytorch_torchdynamo_jit_trace`, `pytorch_torchdynamo_jit_script_ofi`, `pytorch_torchdynamo_jit_trace_ofi` |
| PyTorch | [Intel Neural Compressor (INC) Mixed Precision](https://github.com/intel/neural-compressor/blob/master/docs/source/mixed_precision.md) | `pytorch_inc_bf16` | 
| PyTorch | [INC INT8 Static Quantization (FX/IPEX)](https://github.com/intel/neural-compressor/blob/master/docs/source/PTQ.md) | `pytorch_inc_static_quant_fx`, `pytorch_inc_static_quant_ipex` |
| PyTorch | [INC INT8 Dynamic Quantization](https://github.com/intel/neural-compressor/blob/master/docs/source/dynamic_quantization.md) | `pytorch_inc_dynamic_quant` |
| PyTorch | [Intel Extension for PyTorch (FP32, BF16, INT8 Static/Dynamic Quantization)](https://github.com/intel/intel-extension-for-pytorch) | `pytorch_ipex_fp32`, `pytorch_ipex_bf16`, `pytorch_ipex_int8_static_quant`, `pytorch_ipex_int8_dynamic_quant` |
| PyTorch | [Alibaba Blade-DISC](https://github.com/alibaba/BladeDISC) | `pytorch_aliblade` |
| PyTorch Lightning | [Mixed Precision](https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html) | `pytorch_lightning_bf16_cpu` |
| TensorFlow | [Mixed Precision](https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html) | `tensorflow_amp` |
| Keras | [Mixed Precision](https://www.tensorflow.org/guide/mixed_precision) | `keras_amp` |
| TensorFlow/Keras Model | [INC Quantization](https://github.com/intel/neural-compressor/blob/master/docs/source/PTQ.md) | `tensorflow_inc` |
| Keras Script | [INC Quantization](https://github.com/intel/neural-compressor/tree/master/examples/keras/mnist) | `keras_inc` |
| ONNX Runtime | [INC Static Quantization (QLinear)](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/README.md#operator-oriented-with-qlinearops) | `onnx_inc_static_quant_qlinear` |
| ONNX Runtime | [INC Static Quantization (QDQ)](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/README.md#tensor-oriented-qdq-format) | `onnx_inc_static_quant_qdq` |
| ONNX Runtime | [INC Dynamic Quantization](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/README.md#dynamic-quantization) | `onnx_inc_dynamic_quant` |
| [HuggingFace Optimum-Intel](https://huggingface.co/docs/optimum/intel/index) | INC Quantization | `pytorch_inc_huggingface_optimum_static`, `pytorch_inc_huggingface_optimum_dynamic` |
| [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers/) | INC Quantization | `intel_extension_for_transformers` |
| [BigDL Nano](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Nano/pytorch.html#bigdl-nano-pytorch-inferenceoptimizer) | [Optimization List](./BigDLNanoSupport.md) | `nano_` + [specific alias](./BigDLNanoSupport.md) |
| Auto-Detect | [INC Quantization](https://github.com/intel/neural-compressor) | `inc_auto` |

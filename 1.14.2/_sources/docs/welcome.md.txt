Introduction to Intel® Neural Compressor 
==========================

Intel® Neural Compressor (formerly known as Intel® Low Precision Optimization Tool) is an open-source Python library running on Intel CPUs and GPUs, which delivers unified interfaces across multiple deep learning frameworks for popular network compression technologies, such as quantization, pruning, knowledge distillation. This tool supports automatic accuracy-driven tuning strategies to help user quickly find out the best quantized model. It also implements different weight pruning algorithms to generate pruned model with predefined sparsity goal and supports knowledge distillation to distill the knowledge from the teacher model to the student model.

> **Note**: GPU support is under development.

| Architecture | Workflow |
| - | - |
| ![Architecture](imgs/architecture.png "Architecture") | ![Workflow](imgs/workflow.png "Workflow") |

Supported deep learning frameworks are:

* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [1.15.0 UP3](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up3), [1.15.0 UP2](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up2), [1.15.0 UP1](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up1), [2.1.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.1.0), [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0), [2.4.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.4.0), [2.5.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.5.0), [Official TensorFlow 2.6.0](https://github.com/tensorflow/tensorflow/tree/v2.6.0)

>  **Note**: Intel Optimized TensorFlow 2.5.0 requires setting environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 before running quantization process or deploying the quantized model.

>  **Note**: From the official TensorFlow 2.6.0, oneDNN support has been upstreamed. Download the official TensorFlow 2.6.0 binary for the CPU device and set the environment variable TF_ENABLE_ONEDNN_OPTS=1 before running the quantization process or deploying the quantized model.

* [PyTorch\*](https://pytorch.org/), including [1.5.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.6.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.8.0+cpu](https://download.pytorch.org/whl/torch_stable.html)
* [Apache\* MXNet](https://mxnet.apache.org), including [1.6.0](https://github.com/apache/incubator-mxnet/tree/1.6.0), [1.7.0](https://github.com/apache/incubator-mxnet/tree/1.7.0), [1.8.0](https://github.com/apache/incubator-mxnet/tree/1.8.0)
* [ONNX\* Runtime](https://github.com/microsoft/onnxruntime), including [1.6.0](https://github.com/microsoft/onnxruntime/tree/v1.6.0), [1.7.0](https://github.com/microsoft/onnxruntime/tree/v1.7.0), [1.8.0](https://github.com/microsoft/onnxruntime/tree/v1.8.0)

[Get started](getting_started.md) with installation, tutorials, examples, and more!

View the Intel® Neural Compressor repo at: <https://github.com/intel/neural-compressor>.

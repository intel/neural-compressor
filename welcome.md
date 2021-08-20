Introduction to Intel LPOT
==========================

The Intel® Low Precision Optimization Tool (Intel® LPOT) is an open-source Python library that delivers a unified low-precision inference interface across multiple Intel-optimized Deep Learning (DL) frameworks on both CPUs and GPUs. It supports automatic accuracy-driven tuning strategies, along with additional objectives such as optimizing for performance, model size, and memory footprint. It also provides easy extension capability for new backends, tuning strategies, metrics, and objectives.

> **Note**: GPU support is under development.

| Infrastructure | Workflow |
| - | - |
| ![LPOT Infrastructure](./docs/imgs/infrastructure.png "Infrastructure") | ![LPOT Workflow](./docs/imgs/workflow.png "Workflow") |

Supported Intel-optimized DL frameworks are:

* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [1.15.0 UP3](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up3), [1.15.0 UP2](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up2), [1.15.0 UP1](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up1), [2.1.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.1.0), [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0), [2.4.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.4.0), [2.5.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.5.0), [Official TensorFlow 2.6.0](https://github.com/tensorflow/tensorflow/tree/v2.6.0)

>  **Note**: Intel Optimized TensorFlow 2.5.0 requires to set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 before running LPOT quantization or deploying the quantized model.

>  **Note**: From Official TensorFlow 2.6.0, oneDNN support has been upstreamed. User just need download official TensorFlow binary for CPU device and set environment variable TF_ENABLE_ONEDNN_OPTS=1 before running LPOT quantization or deploying the quantized model.

* [PyTorch\*](https://pytorch.org/), including [1.5.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.6.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.8.0+cpu](https://download.pytorch.org/whl/torch_stable.html)
* [Apache\* MXNet](https://mxnet.apache.org), including [1.6.0](https://github.com/apache/incubator-mxnet/tree/1.6.0), [1.7.0](https://github.com/apache/incubator-mxnet/tree/1.7.0), [1.8.0](https://github.com/apache/incubator-mxnet/tree/1.8.0)
* [ONNX\* Runtime](https://github.com/microsoft/onnxruntime), including [1.6.0](https://github.com/microsoft/onnxruntime/tree/v1.6.0), [1.7.0](https://github.com/microsoft/onnxruntime/tree/v1.7.0), [1.8.0](https://github.com/microsoft/onnxruntime/tree/v1.8.0)

[Get started](getting_started.md) with installation, tutorials, examples, and more!

View the Intel® LPOT repo at: <https://github.com/intel/lpot>.

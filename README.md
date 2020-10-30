Intel® Low Precision Optimization Tool
=========================================

Intel® Low Precision Optimization Tool is an open-source python library which is intended to deliver a unified low-precision inference interface cross multiple Intel optimized DL frameworks on both CPU and GPU. It supports automatic accuracy-driven tuning strategies, along with additional objectives like optimizing for performance, model size and memory footprint. It also provides the easy extension capability for new backends, tuning strategies, metrics and objectives.


> **WARNING**
>
> GPU support is under development.

Supported Intel optimized DL frameworks are:
* [Tensorflow\*](https://www.tensorflow.org)
* [PyTorch\*](https://pytorch.org/)
* [Apache\* MXNet](https://mxnet.apache.org)

Supported tuning strategies are:
* [Basic](docs/introduction.md#basic-strategy)
* [Bayesian](docs/introduction.md#bayesian-strategy)
* [MSE](docs/introduction.md#mse-strategy)
* [Exhaustive](docs/introduction.md#exhaustive-strategy)
* [Random](docs/introduction.md#random-strategy)
* [TPE](docs/tuning_strategy.md#TPE-strategy)

Mixed precision support:
* [int8](docs/mixed_precision.md#int8)
* [BFP16](docs/mixed_precision.md#BFP16)


# Introduction 

  [Introduction](docs/introduction.md) explains Intel® Low Precision Optimization Tool infrastructure, design philosophy, supported functionality, details of tuning strategy implementations and tuning result on popular models.

# Tutorials
* [Hello World](examples/helloworld/README.md) demonstrates the simple steps to utilize Intel® Low Precision Optimization Tool for quanitzation, which can help you quick start with the tool.
* [Tutorials](docs/README.md) provides comprehensive instructions of how to utilize diffrennt features of Intel® Low Precision Optimization Tool.
* [Features](docs/index.md) provides the introduction of features such as tuning strategy, QAT, pruning and so on.
* [Examples](examples) is a tuning zoo to demonstrate the usage of Intel® Low Precision Optimization Tool in TensorFlow, PyTorch and MxNet for industry models of diffrent categories.  

# Install from source 

  ```Shell
  git clone https://github.com/intel/lp-opt-tool.git
  cd lp-opt-tool
  python setup.py install
  ```

# Install from binary

  ```Shell
  # install from pip
  pip install ilit

  # install from conda
  conda config --add channels intel
  conda install ilit
  ```

# System Requirements

### Hardware

Intel® Low Precision Optimization Tool supports systems based on Intel 64 architecture or compatible processors.

### Software

Intel® Low Precision Optimization Tool requires to install Intel optimized framework version for TensorFlow, PyTorch, and MXNet.

# Tuning Zoo

The followings are the examples integrated with Intel® Low Precision Optimization Tool for auto tuning.

| TensorFlow Model                                                    | Category  |
|---------------------------------------------------------------------|------------|
|[ResNet50 V1](examples/tensorflow/image_recognition/README.md)        | Image Recognition |
|[ResNet50 V1.5](examples/tensorflow/image_recognition/README.md)      | Image Recognition |
|[ResNet101](examples/tensorflow/image_recognition/README.md)          | Image Recognition |
|[Inception V1](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[Inception V2](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[Inception V3](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[Inception V4](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[ResNetV2_50](examples/tensorflow/image_recognition/README.md)        | Image Recognition |
|[ResNetV2_101](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[ResNetV2_152](examples/tensorflow/image_recognition/README.md)       | Image Recognition |
|[Inception ResNet V2](examples/tensorflow/image_recognition/README.md)| Image Recognition |
|[SSD ResNet50 V1](examples/tensorflow/object_detection/README.md)     | Object Detection  |
|[Wide & Deep](examples/tensorflow/recommendation/wide_deep_large_ds/WND_README.md) | Recommendation |
|[VGG16](examples/tensorflow/image_recognition/README.md)              | Image Recognition |
|[VGG19](examples/tensorflow/image_recognition/README.md)              | Image Recognition |
|[Style_transfer](examples/tensorflow/style_transfer/README.md)        | Style Transfer    |


| PyTorch Model                                                               | Category  |
|---------------------------------------------------------------------|------------|
|[BERT-Large RTE](examples/pytorch/language_translation/README.md)  | Language Translation   |
|[BERT-Large QNLI](examples/pytorch/language_translation/README.md) | Language Translation   |
|[BERT-Large CoLA](examples/pytorch/language_translation/README.md) | Language Translation   |
|[BERT-Base SST-2](examples/pytorch/language_translation/README.md) | Language Translation   |
|[BERT-Base RTE](examples/pytorch/language_translation/README.md)   | Language Translation   |
|[BERT-Base STS-B](examples/pytorch/language_translation/README.md) | Language Translation   |
|[BERT-Base CoLA](examples/pytorch/language_translation/README.md)  | Language Translation   |
|[BERT-Base MRPC](examples/pytorch/language_translation/README.md)  | Language Translation   |
|[DLRM](examples/pytorch/recommendation/README.md)                  | Recommendation   |
|[BERT-Large MRPC](examples/pytorch/language_translation/README.md) | Language Translation   |
|[ResNext101_32x8d](examples/pytorch/image_recognition/imagenet/cpu/PTQ/README.md)          | Image Recognition   |
|[BERT-Large SQUAD](examples/pytorch/language_translation/README.md)                | Language Translation   |
|[ResNet50 V1.5](examples/pytorch/image_recognition/imagenet/cpu/PTQ/README.md)             | Image Recognition   |
|[ResNet18](examples/pytorch/image_recognition/imagenet/cpu/PTQ/README.md)                  | Image Recognition   |
|[Inception V3](examples/pytorch/image_recognition/imagenet/cpu/PTQ/README.md)              | Image Recognition   |
|[YOLO V3](examples/pytorch/object_detection/yolo_v3/README.md)                     | Object Detection   |
|[Peleenet](examples/pytorch/image_recognition/peleenet/README.md)                  | Image Recognition   |
|[ResNest50](examples/pytorch/image_recognition/resnest/README.md)                  | Image Recognition   |
|[SE_ResNext50_32x4d](examples/pytorch/image_recognition/se_resnext/README.md)      | Image Recognition   |
|[ResNet50 V1.5 QAT](examples/pytorch/image_recognition/imagenet/cpu/QAT/README.md)     | Image Recognition   |
|[ResNet18 QAT](examples/pytorch/image_recognition/imagenet/cpu/QAT/README.md)          | Image Recognition   |

| MxNet Model                                                               | Category  |
|---------------------------------------------------------------------|------------|
|[ResNet50 V1](examples/mxnet/image_recognition/README.md)           | Image Recognition      |
|[MobileNet V1](examples/mxnet/image_recognition/README.md)          | Image Recognition      |
|[MobileNet V2](examples/mxnet/image_recognition/README.md)          | Image Recognition      |
|[SSD-ResNet50](examples/mxnet/object_detection/README.md)           | Object Detection       |
|[SqueezeNet V1](examples/mxnet/image_recognition/README.md)         | Image Recognition      |
|[ResNet18](examples/mxnet/image_recognition/README.md)              | Image Recognition      |
|[Inception V3](examples/mxnet/image_recognition/README.md)          | Image Recognition      |


# Known Issues

1. KL Divergence Algorithm is very slow at TensorFlow

   Due to TensorFlow not supporting tensor dump naturally, current solution of dumping the tensor content is adding print op and dumping the value to stdout. So if the model to tune is a TensorFlow model, please restrict calibration.algorithm.activation and calibration.algorithm.weight in user YAML config file to minmax.

2. MSE tuning strategy doesn't work with PyTorch adaptor layer

   MSE tuning strategy requires to compare FP32 tensor and INT8 tensor to decide which op has impact on final quantization accuracy. PyTorch adaptor layer doesn't implement this inspect tensor interface. So if the model to tune is a PyTorch model, please do not choose MSE tuning strategy.

# Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/lp-opt-tool/issues) page. You may also reach out to ilit.maintainers@intel.com.

# Contributing

We welcome community contributions to Intel® Low Precision Optimization Tool. If you have an idea on how
to improve the library:

* For changes impacting the public API, submit
  an [RFC pull request](CONTRIBUTING.md#RFC_pull_requests).
* Ensure that the changes are consistent with the
 [code contribution guidelines](CONTRIBUTING.md#code_contribution_guidelines)
 and [coding style](CONTRIBUTING.md#coding_style).
* Ensure that you can run all the examples with your patch.
* Submit a [pull request](https://github.com/intel/lp-opt-tool/pulls).

For additional details, see [contribution guidelines](CONTRIBUTING.md).

This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

# License

Intel® Low Precision Optimization Tool is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  This
software includes components with separate copyright notices and license
terms. Your use of the source code for these components is subject to the terms
and conditions of the following licenses.

Apache License Version 2.0:
* [Intel TensorFlow Quantization Tool](https://github.com/IntelAI/tools)

MIT License:
* [bayesian-optimization](https://github.com/fmfn/BayesianOptimization)

See accompanying [LICENSE](LICENSE) file for full license text and copyright notices.

--------

[Legal Information](legal_information.md)

## Citing

If you use Intel® Low Precision Optimization Tool in your research or wish to refer to the tuning results published in the [Tuning Zoo](#tuning-zoo), please use the following BibTeX entry.

```
@misc{Intel® Low Precision Optimization Tool,
  author =       {Feng Tian, Chuanqi Wang, Guoming Zhang, Penghui Cheng, Pengxin Yuan, Haihao Shen, and Jiong Gong},
  title =        {Intel® Low Precision Optimization Tool},
  howpublished = {\url{https://github.com/intel/lp-opt-tool}},
  year =         {2020}
}
```

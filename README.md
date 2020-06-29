Intel Low Precision Inference Tool (iLiT)
=========================================

Intel Low Precision Inference Tool (iLiT) is an open-source python library which is intended to deliver a unified low-precision inference interface cross multiple Intel optimized DL frameworks on both CPU and GPU. It supports automatic accuracy-driven tuning strategies, along with additional bjectives like performance, model size, or memory footprint. It also provides the easy extension capability for new backends, tuning strategies, metrics and objectives.

> **WARNING**
>
> GPU support is under development.

Currently supported Intel optimized DL frameworks are:
* [Tensorflow\*](https://www.tensorflow.org)
* [PyTorch\*](https://pytorch.org/)
* [Apache\* MXNet](https://mxnet.apache.org)

Currently supported tuning strategies are:
* [Basic](docs/introduction.md#basic-strategy)
* [Random](docs/introduction.md#random-strategy)
* [Exhaustive](docs/introduction.md#exhaustive-strategy)
* [Bayesian](docs/introduction.md#bayesian-strategy)
* [MSE](docs/introduction.md#mse-strategy)


# Documentation

* [Introduction](docs/introduction.md) explains iLiT infrastructure, design philosophy, supported functionality, details of tuning strategy implementations and tuning result on popular models.
* [Tutorial](docs/tutorial.md) provides
comprehensive step-by-step instructions of how to enable iLiT on sample models.

# Install from source 

  ```Shell
  git clone https://github.com/intel/lp-inference-kit.git
  cd lp-inference-kit
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

iLiT supports systems based on Intel 64 architecture or compatible processors.

### Software

iLiT requires to install Intel optimized framework version for TensorFlow, PyTorch, and MXNet.

# Tuning Zoo

The followings are the examples integrated with iLiT for auto tuning.

| Model                                                     | Framework | Model                                                   | Framework | Model                                                                  | Framework  |
|-----------------------------------------------------------|-----------|---------------------------------------------------------|-----------|------------------------------------------------------------------------|------------|
| [ResNet50 V1](examples/mxnet/cnn/README.md)               | MXNet     | [BERT-Large RTE](examples/pytorch/bert/BERT_README.md)  | PyTorch   | [ResNet18](examples/pytorch/resnet/README.md)                          | PyTorch    |
| [MobileNet V1](examples/mxnet/cnn/README.md)              | MXNet     | [BERT-Large QNLI](examples/pytorch/bert/BERT_README.md) | PyTorch   | [ResNet50 V1](examples/tensorflow/image_recognition/README.md)         | TensorFlow |
| [MobileNet V2](examples/mxnet/cnn/README.md)              | MXNet     | [BERT-Large CoLA](examples/pytorch/bert/BERT_README.md) | PyTorch   | [ResNet50 V1.5](examples/tensorflow/image_recognition/README.md)       | TensorFlow |
| [SSD-ResNet50](examples/mxnet/object_detection/README.md) | MXNet     | [BERT-Base SST-2](examples/pytorch/bert/BERT_README.md) | PyTorch   | [ResNet101](examples/tensorflow/image_recognition/README.md)           | TensorFlow |
| [SqueezeNet V1](examples/mxnet/cnn/README.md)             | MXNet     | [BERT-Base RTE](examples/pytorch/bert/BERT_README.md)   | PyTorch   | [Inception V1](examples/tensorflow/image_recognition/README.md)        | TensorFlow |
| [ResNet18](examples/mxnet/cnn/README.md)                  | MXNet     | [BERT-Base STS-B](examples/pytorch/bert/BERT_README.md) | PyTorch   | [Inception V2](examples/tensorflow/image_recognition/README.md)        | TensorFlow |
| [Inception V3](examples/mxnet/cnn/README.md)              | MXNet     | [BERT-Base CoLA](examples/pytorch/bert/BERT_README.md)  | PyTorch   | [Inception V3](examples/tensorflow/image_recognition/README.md)        | TensorFlow |
| [DLRM](examples/pytorch/dlrm/DLRM_README.md)              | PyTorch   | [BERT-Base MRPC](examples/pytorch/bert/BERT_README.md)  | PyTorch   | [Inception V4](examples/tensorflow/image_recognition/README.md)        | TensorFlow |
| [BERT-Large MRPC](examples/pytorch/dlrm/DLRM_README.md)   | PyTorch   | [ResNet101](examples/pytorch/resnet/README.md)          | PyTorch   | [Inception ResNet V2](examples/tensorflow/image_recognition/README.md) | TensorFlow |
| [BERT-Large SQUAD](examples/pytorch/bert/BERT_README.md)  | PyTorch   | [ResNet50 V1.5](examples/pytorch/resnet/README.md)      | PyTorch   | [SSD ResNet50 V1](examples/tensorflow/object_detection/README.md)      | TensorFlow |


# Known Issues

1. KL Divergence Algorithm is very slow at TensorFlow

   Due to TensorFlow not supporting tensor dump naturally, current solution of dumping the tensor content is adding print op and dumpping the value to stdout. So if the model to tune is a TensorFlow model, please restrict calibration.algorithm.activation and calibration.algorithm.weight in user yaml config file to minmax.

2. MSE tuning strategy doesn't work with PyTorch adaptor layer

   MSE tuning strategy requires to compare FP32 tensor and INT8 tensor to decide which op has impact on final quantization accuracy. PyTorch adaptor layer doesn't implement this inspect tensor interface. So if the model to tune is a PyTorch model, please not choose MSE tuning strategy.

# Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/lp-inference-kit/issues) page. You may also reach out to ilit.maintainers@intel.com.

# Contributing

We welcome community contributions to iLiT. If you have an idea on how
to improve the library:

* For changes impacting the public API, submit
  an [RFC pull request](CONTRIBUTING.md#RFC_pull_requests).
* Ensure that the changes are consistent with the
 [code contribution guidelines](CONTRIBUTING.md#code_contribution_guidelines)
 and [coding style](CONTRIBUTING.md#coding_style).
* Ensure that you can run all the examples with your patch.
* Submit a [pull request](https://github.com/intel/lp-inference-kit/pulls).

For additional details, see [contribution guidelines](CONTRIBUTING.md).

This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

# License

iLiT is licensed under
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

If you use iLiT in your research or wish to refer to the tuning results published in the [Tuning Zoo](#tuning-zoo), please use the following BibTeX entry.

```
@misc{iLiT,
  author =       {Feng Tian, Chuanqi Wang, Guoming Zhang, Penghui Cheng, Pengxin Yuan, Haihao Shen, and Jiong Gong},
  title =        {Intel Low Precision Inference Tool},
  howpublished = {\url{https://github.com/intel/lp-inference-kit}},
  year =         {2020}
}
```

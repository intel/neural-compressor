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
* [Basic](Introduction.md#Basic\ Strategy)
* [Random](Introduction.md#Random\ Strategy)
* [Exhaustive](Introduction.md#Exhaustive\ Strategy)
* [Bayesian](Introduction.md#Bayesian\ Strategy)
* [MSE](Introduction.md#MSE\ Strategy)


# Documentation

* [Introduction](Introduction.md) explains iLiT infrastructure, design philosophy, supported functionality, details of tuning strategy implementations and tuning result on popular models.
* [Tutorial](Tutorial.md) provides
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

Below table is the tuning result using iLiT.

| Model               | Framework  | Tuning Strategy | FP32 Accuracy Baseline | INT8 Tuning Accuracy | FP32/INT8 Perf Ratio |
|---------------------|------------|-----------------|------------------------|----------------------|----------------------|
| BERT-Base MRPC      | MXNet      |                 |                        |                      |                      |
| BERT-Base SQUAD     | MXNet      |                 |                        |                      |                      |
| ResNet50 V1         | MXNet      |                 |                        |                      |                      |
| MobileNet V1        | MXNet      |                 |                        |                      |                      |
| MobileNet V2        | MXNet      |                 |                        |                      |                      |
| SSD-MobileNet V1    | MXNet      |                 |                        |                      |                      |
| SSD-ResNet50        | MXNet      |                 |                        |                      |                      |
| SqueezeNet V1       | MXNet      |                 |                        |                      |                      |
| ResNet18            | MXNet      |                 |                        |                      |                      |
| Inception V3        | MXNet      |                 |                        |                      |                      |
| DLRM                | PyTorch    |                 |                        |                      |                      |
| BERT-Large MRPC     | PyTorch    |                 |                        |                      |                      |
| BERT-Large SQUAD    | PyTorch    |                 |                        |                      |                      |
| BERT-Large CoLA     | PyTorch    |                 |                        |                      |                      |
| BERT-Base STS-B     | PyTorch    |                 |                        |                      |                      |
| BERT-Base CoLA      | PyTorch    |                 |                        |                      |                      |
| BERT-Base MRPC      | PyTorch    |                 |                        |                      |                      |
| BERT-Base SST-2     | PyTorch    |                 |                        |                      |                      |
| BERT-Base RTE       | PyTorch    |                 |                        |                      |                      |
| BERT-Large RTE      | PyTorch    |                 |                        |                      |                      |
| BERT-Large QNLI     | PyTorch    |                 |                        |                      |                      |
| ResNet50 V1.5       | PyTorch    |                 |                        |                      |                      |
| ResNet18            | PyTorch    |                 |                        |                      |                      |
| ResNet101           | PyTorch    |                 |                        |                      |                      |
| ResNet50 V1         | TensorFlow |                 |                        |                      |                      |
| ResNet50 V1.5       | TensorFlow |                 |                        |                      |                      |
| ResNet101           | TensorFlow |                 |                        |                      |                      |
| MobileNet V1        | TensorFlow |                 |                        |                      |                      |
| MobileNet V2        | TensorFlow |                 |                        |                      |                      |
| Inception V1        | TensorFlow |                 |                        |                      |                      |
| Inception V2        | TensorFlow |                 |                        |                      |                      |
| Inception V3        | TensorFlow |                 |                        |                      |                      |
| Inception V4        | TensorFlow |                 |                        |                      |                      |
| Inception ResNet V2 | TensorFlow |                 |                        |                      |                      |

# Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/lp-inference-kit/issues) page.

You may reach out to [iLiT Maintainers](ilit.maintainers@intel.com).

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

If you use iLiT in your research or wish to refer to the tuning results published in the [Tuning Zoo](#Tuning-Zoo)), please use the following BibTeX entry.

```
@misc{iLiT,
  author =       {Feng Tian and Chuanqi Wang and Guoming Zhang and
                  Penghui Cheng and Pengxin Yuan and Haihao Shen and Jiong Gong},
  title =        {Intel Low Precision Inference Tool},
  howpublished = {\url{https://github.com/intel/lp-inference-kit}},
  year =         {2020}
}
```

Intel® Low Precision Optimization Tool
======================================

Intel® Low Precision Optimization Tool (Intel® LPOT) is an open-source Python library that delivers a unified low-precision inference interface across multiple Intel-optimized DL frameworks on both CPUs and GPUs. It supports automatic accuracy-driven tuning strategies, along with additional objectives such as optimizing for performance, model size, and memory footprint. It also provides easy extension capability for new backends, tuning strategies, metrics, and objectives.

> **Note**
>
> GPU support is under development.

<table>
  <tr>
    <td>Infrastructure</td>
    <td>Workflow</td>
  </tr>
  <tr>
    <td><img src="docs/imgs/infrastructure.jpg" width=640 height=320></td>
    <td><img src="docs/imgs/workflow.jpg" width=640 height=320></td>
  </tr>
 </table>

Supported Intel optimized DL frameworks are:
* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [1.15.0 UP2](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up2), [1.15.0 UP1](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up1), [2.1.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.1.0), [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0)
* [PyTorch\*](https://pytorch.org/), including [1.5.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.6.0+cpu](https://download.pytorch.org/whl/torch_stable.html)
* [Apache\* MXNet](https://mxnet.apache.org), including [1.6.0](https://github.com/apache/incubator-mxnet/tree/1.6.0), [1.7.0](https://github.com/apache/incubator-mxnet/tree/1.7.0)
* [ONNX\* Runtime](https://github.com/microsoft/onnxruntime), including [1.6.0](https://github.com/microsoft/onnxruntime/tree/v1.6.0)


# Installation
## Install for Linux
### Install from binary

  ```Shell
  # install from pip
  pip install lpot

  # install from conda
  conda install lpot -c conda-forge -c intel 
  ```

### Install from source

  ```Shell
  git clone https://github.com/intel/lpot.git
  cd lpot
  python setup.py install
  ```

## Install for Windows
### Install from binary
  ```Shell
  # install from pip
  pip install lpot

  # install from conda
  conda install lpot -c conda-forge -c intel 
  ```
### Install from source

#### **Prerequisites**

The following prerequisites and requirements must be satisfied in order to install successfully：

- Python version: 3.6 or 3.7 or 3.8

- Download and install anaconda: [anaconda](https://anaconda.org/)

- Create a virtual environment named lpot in anaconda:
    ```shell
    # Here we install python 3.7 for instance. You can also choose python 3.6 & 3.8.
    conda create -n lpot python=3.7
    conda activate lpot
    ```

#### **Installation Procedure**

```shell
git clone https://github.com/intel/lpot.git
cd lpot
pip install -r requirements.txt
python setup.py install
```

# Getting started

* [Introduction](docs/introduction.md) explains Intel® Low Precision Optimization Tool's API.
* [Quantization](docs/backend_quant.md) explains quantization support matrix on multiple frameworks.
* [Tutorial](docs/tutorial.md) provides comprehensive instructions on how to utilize Intel® Low Precision Optimization Tool's features with examples. 
* [Examples](examples) are provided to demonstrate the usage of Intel® Low Precision Optimization Tool in different frameworks: [TensorFlow](examples/tensorflow), [PyTorch](examples/pytorch), [MXNet](examples/mxnet) and [ONNX Runtime](examples/onnxrt).


# Deep Dive

* [Quantization](docs/Quantization.md) is the processes that enable inference and training by performing computations at low precision data type, such as fixed point integers. LPOT supports [Post-Training Quantization](docs/PTQ.md) and [Quantization-Aware Training](docs/QAT.md)
* [Pruning](docs/pruning.md) provides a common method for introducing sparsity in weights and activations.
* [Benchmarking](docs/benchmark.md) introduces how to utilize the benchmark interface of LPOT.
* [Mixed precision](docs/mixed_precision.md) introduces how to enable mixed precision, including BFP16 and int8 and FP32, on Intel platforms during tuning.
* [Transform](docs/transform.md) introduces how to utilize LPOT buildin data processing and how to develop a custom data processing method. 
* [Dataset](docs/dataset.md) introudces how to utilize LPOT buildin dataset and how to develop a custom dataset.
* [Metric](docs/metric.md) introduces how to utilize LPOT buildin metric and how to develop a custom metric.
* [TensorBoard](docs/tensorboard.md) provides tensor histogram and execution graph for tuning debugging purpose.
* [PyTorch Deploy](docs/pytorch_model_saving.md) introduces how LPOT saves and loads quantized PyTorch model.


# Advanced Topics

* [Adaptor](docs/adaptor.md) is the interface between LPOT and framework. The method to develop adaptor extension is introduced with ONNX Runtime as example. 
* [Strategy](docs/tuning_strategies.md) can automatically optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria. The method to develop a new strategy is introduced.


# System Requirements

Intel® Low Precision Optimization Tool supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64), specially optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, and Cooper Lake)
* future Intel Xeon Scalable processor (code name Sapphire Rapids)

Intel® Low Precision Optimization Tool requires installing the pertinent Intel-optimized framework version for TensorFlow, PyTorch, and MXNet.

### Validated Hardware/Software Environment

<table>
<thead>
  <tr>
    <th class="tg-bobw">Platform</th>
    <th class="tg-bobw">OS</th>
    <th class="tg-bobw">Python</th>
    <th class="tg-bobw">Framework</th>
    <th class="tg-bobw">Version</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="10">Cascade Lake<br><br>Cooper Lake<br><br>Skylake</td>
    <td class="tg-nrix" rowspan="10">CentOS 7.8<br><br>Ubuntu 18.04</td>
    <td class="tg-nrix" rowspan="10">3.6<br><br>3.7</td>
    <td class="tg-cly1" rowspan="6">TensorFlow</td>
    <td class="tg-7zrl">2.2.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP1</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP2</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.3.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.1.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.2</td>
  </tr>
  <tr>
    <td class="tg-7zrl">PyTorch</td>
    <td class="tg-7zrl">1.5.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-cly1" rowspan="2">MXNet</td>
    <td class="tg-7zrl">1.7.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ONNX Runtime</td>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
</tbody>
</table>

# Model Zoo

Intel® Low Precision Optimization Tool provides numerous examples to show promising accuracy loss with the best performance gain.

<table>
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2">Framework</th>
    <th class="tg-9wq8" rowspan="2">Version</th>
    <th class="tg-9wq8" rowspan="2">Model</th>
    <th class="tg-9wq8" rowspan="2">Dataset</th>
    <th class="tg-pb0m" colspan="3">TOP-1   Accuracy</th>
    <th class="tg-za14">Performance Speedup</th>
  </tr>
  <tr>
    <td class="tg-za14">INT8 Tuning Accuracy</td>
    <td class="tg-za14">FP32 Accuracy   Baseline</td>
    <td class="tg-za14">Acc   Ratio[(INT8-FP32)/FP32]</td>
    <td class="tg-za14">Real-time Latency Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8" rowspan="20">2.2.0</td>
    <td class="tg-9wq8">resnet50v1.0</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">73.80%</td>
    <td class="tg-za14">74.30%</td>
    <td class="tg-9wq8">-0.67%</td>
    <td class="tg-9wq8">2.25x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">resnet50v1.5</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">76.80%</td>
    <td class="tg-za14">76.50%</td>
    <td class="tg-9wq8">0.39%</td>
    <td class="tg-9wq8">2.32x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">resnet101</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">77.20%</td>
    <td class="tg-za14">76.40%</td>
    <td class="tg-9wq8">1.05%</td>
    <td class="tg-9wq8">2.75x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">inception_v1</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">70.10%</td>
    <td class="tg-za14">69.70%</td>
    <td class="tg-9wq8">0.57%</td>
    <td class="tg-9wq8">1.56x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">inception_v2</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">74.00%</td>
    <td class="tg-za14">74.00%</td>
    <td class="tg-9wq8">0.00%</td>
    <td class="tg-9wq8">1.68x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">inception_v3</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">77.20%</td>
    <td class="tg-za14">76.70%</td>
    <td class="tg-9wq8">0.65%</td>
    <td class="tg-9wq8">2.05x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">inception_v4</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">80.00%</td>
    <td class="tg-za14">80.30%</td>
    <td class="tg-9wq8">-0.37%</td>
    <td class="tg-9wq8">2.52x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">inception_resnet_v2</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">80.20%</td>
    <td class="tg-za14">80.40%</td>
    <td class="tg-9wq8">-0.25%</td>
    <td class="tg-9wq8">1.75x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">mobilenetv1</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">71.10%</td>
    <td class="tg-za14">71.00%</td>
    <td class="tg-9wq8">0.14%</td>
    <td class="tg-9wq8">1.88x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">ssd_resnet50_v1</td>
    <td class="tg-9wq8">Coco</td>
    <td class="tg-za14">37.72%</td>
    <td class="tg-za14">38.01%</td>
    <td class="tg-9wq8">-0.76%</td>
    <td class="tg-9wq8">2.88x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">mask_rcnn_inception_v2</td>
    <td class="tg-9wq8">Coco</td>
    <td class="tg-za14">28.75%</td>
    <td class="tg-za14">29.13%</td>
    <td class="tg-9wq8">-1.30%</td>
    <td class="tg-9wq8">4.14x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">wide_deep_large_ds</td>
    <td class="tg-c3ow">criteo-kaggle</td>
    <td class="tg-za14">77.61%</td>
    <td class="tg-za14">77.67%</td>
    <td class="tg-9wq8">-0.08%</td>
    <td class="tg-9wq8">1.41x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">vgg16</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">72.10%</td>
    <td class="tg-za14">70.90%</td>
    <td class="tg-9wq8">1.69%</td>
    <td class="tg-9wq8">3.71x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">vgg19</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">72.30%</td>
    <td class="tg-za14">71.00%</td>
    <td class="tg-9wq8">1.83%</td>
    <td class="tg-9wq8">3.78x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">resnetv2_50</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">70.20%</td>
    <td class="tg-za14">69.60%</td>
    <td class="tg-9wq8">0.86%</td>
    <td class="tg-9wq8">1.52x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">resnetv2_101</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">72.50%</td>
    <td class="tg-za14">71.90%</td>
    <td class="tg-9wq8">0.83%</td>
    <td class="tg-9wq8">1.59x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">resnetv2_152</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">72.70%</td>
    <td class="tg-za14">72.40%</td>
    <td class="tg-9wq8">0.41%</td>
    <td class="tg-9wq8">1.62x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">densenet121</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">72.60%</td>
    <td class="tg-za14">72.90%</td>
    <td class="tg-9wq8">-0.41%</td>
    <td class="tg-9wq8">1.84x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">densenet161</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">76.10%</td>
    <td class="tg-za14">76.30%</td>
    <td class="tg-9wq8">-0.26%</td>
    <td class="tg-9wq8">1.44x</td>
  </tr>
  <tr>
    <td class="tg-9wq8">TensorFlow</td>
    <td class="tg-9wq8">densenet169</td>
    <td class="tg-9wq8">ImageNet</td>
    <td class="tg-za14">74.40%</td>
    <td class="tg-za14">74.60%</td>
    <td class="tg-9wq8">-0.27%</td>
    <td class="tg-9wq8">1.22x</td>
  </tr>
</tbody>
</table>


<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th colspan="3">TOP-1 Accuracy</th>
    <th>Performance Speedup</th>
  </tr>
  <tr>
    <td>INT8 Tuning Accuracy</td>
    <td>FP32 Accuracy Baseline</td>
    <td>Acc Ratio[(INT8-FP32)/FP32]</td>
    <td>Real-time Latency Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MXNet</td>
    <td rowspan="9">1.7.0</td>
    <td>resnet50v1</td>
    <td>ImageNet</td>
    <td>76.03%</td>
    <td>76.33%</td>
    <td>-0.39%</td>
    <td>3.18x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>inceptionv3</td>
    <td>ImageNet</td>
    <td>77.80%</td>
    <td>77.64%</td>
    <td>0.21%</td>
    <td>2.65x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>mobilenet1.0</td>
    <td>ImageNet</td>
    <td>71.72%</td>
    <td>72.22%</td>
    <td>-0.69%</td>
    <td>2.62x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>mobilenetv2_1.0</td>
    <td>ImageNet</td>
    <td>70.77%</td>
    <td>70.87%</td>
    <td>-0.14%</td>
    <td>2.89x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>resnet18_v1</td>
    <td>ImageNet</td>
    <td>69.99%</td>
    <td>70.14%</td>
    <td>-0.21%</td>
    <td>3.08x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>squeezenet1.0</td>
    <td>ImageNet</td>
    <td>56.88%</td>
    <td>56.96%</td>
    <td>-0.14%</td>
    <td>2.55x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>ssd-resnet50_v1</td>
    <td>VOC</td>
    <td>80.21%</td>
    <td>80.23%</td>
    <td>-0.02%</td>
    <td>4.16x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>ssd-mobilenet1.0</td>
    <td>VOC</td>
    <td>74.94%</td>
    <td>75.54%</td>
    <td>-0.79%</td>
    <td>3.31x</td>
  </tr>
  <tr>
    <td>MXNet</td>
    <td>resnet152_v1</td>
    <td>ImageNet</td>
    <td>78.32%</td>
    <td>78.54%</td>
    <td>-0.28%</td>
    <td>3.16x</td>
  </tr>
</tbody>
</table>


# Known Issues

The MSE tuning strategy does not work with the PyTorch adaptor layer. This strategy requires a comparison between the FP32 and INT8 tensors to decide which op impacts the final quantization accuracy. The PyTorch adaptor layer does not implement this inspect tensor interface. Therefore, do not choose the MSE tuning strategy for PyTorch models.

# Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/intel/lpot/issues) page. You may also reach out to lpot.maintainers@intel.com.

# Contribution

We welcome community contributions to Intel® Low Precision Optimization
Tool. If you have an idea on how to improve the library, refer to the following:

* For changes impacting the public API, submit an [RFC pull request](CONTRIBUTING.md#RFC_pull_requests).
* Ensure that the changes are consistent with the [code contribution guidelines](CONTRIBUTING.md#code_contribution_guidelines) and [coding style](CONTRIBUTING.md#coding_style).
* Ensure that you can run all the examples with your patch.
* Submit a [pull request](https://github.com/intel/lpot/pulls).

For additional details, see [contribution guidelines](CONTRIBUTING.md).

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

# License

Intel® Low Precision Optimization Tool is licensed under [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). This software includes components that have separate copyright notices and licensing terms. Your use of the source code for these components is subject to the terms and conditions of the following licenses.

Apache License Version 2.0:
* [Intel TensorFlow Quantization Tool](https://github.com/IntelAI/tools)

MIT License:
* [bayesian-optimization](https://github.com/fmfn/BayesianOptimization)

See the accompanying [LICENSE](LICENSE) file for full license text and copyright notices.

--------

View [Legal Information](legal_information.md).

## Citation

If you use Intel® Low Precision Optimization Tool in your research or you wish to refer to the tuning results published in the [Model Zoo](#model-zoo), use the following BibTeX entry.

```
@misc{Intel® Low Precision Optimization Tool,
  author =       {Feng Tian, Chuanqi Wang, Guoming Zhang, Penghui Cheng, Pengxin Yuan, Haihao Shen, and Jiong Gong},
  title =        {Intel® Low Precision Optimization Tool},
  howpublished = {\url{https://github.com/intel/lpot}},
  year =         {2020}
}
```


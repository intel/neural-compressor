Introduction to Intel® LPOT 
===========================

The Intel® Low Precision Optimization Tool (Intel® LPOT) is an open-source Python library that delivers a unified low-precision inference interface across multiple Intel-optimized Deep Learning (DL) frameworks on both CPUs and GPUs. It supports automatic accuracy-driven tuning strategies, along with additional objectives such as optimizing for performance, model size, and memory footprint. It also provides easy extension capability for new backends, tuning strategies, metrics, and objectives.

> **Note**
>
> GPU support is under development.

**Visit the Intel® LPOT online document website at: <https://intel.github.io/lpot>.**

## Architecture

Intel® LPOT features an infrastructure and workflow that aids in increasing performance and faster deployments across architectures. 


#### Infrastructure

<a target="_blank" href="docs/imgs/infrastructure.png">
  <img src="docs/imgs/infrastructure.png" alt="Infrastructure" width=800 height=360>
</a>

Click the image to enlarge it.

#### Workflow

<a target="_blank" href="docs/imgs/workflow.png">
  <img src="docs/imgs/workflow.png" alt="Workflow" width=800 height=360>
</a>

Click the image to enlarge it.

#### Supported Frameworks

Supported Intel-optimized DL frameworks are:
* [TensorFlow\*](https://github.com/Intel-tensorflow/tensorflow), including [1.15.0 UP3](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up3), [1.15.0 UP2](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up2), [1.15.0 UP1](https://github.com/Intel-tensorflow/tensorflow/tree/v1.15.0up1), [2.1.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.1.0), [2.2.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.2.0), [2.3.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.3.0), [2.4.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.4.0), [2.5.0](https://github.com/Intel-tensorflow/tensorflow/tree/v2.5.0), [Official TensorFlow 2.6.0](https://github.com/tensorflow/tensorflow/tree/v2.6.0)

>  **Note**: Intel Optimized TensorFlow 2.5.0 requires to set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 before running LPOT quantization or deploying the quantized model.

>  **Note**: From Official TensorFlow 2.6.0, oneDNN support has been upstreamed. User just need download official TensorFlow binary for CPU device and set environment variable TF_ENABLE_ONEDNN_OPTS=1 before running LPOT quantization or deploying the quantized model.

* [PyTorch\*](https://pytorch.org/), including [1.5.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.6.0+cpu](https://download.pytorch.org/whl/torch_stable.html), [1.8.0+cpu](https://download.pytorch.org/whl/torch_stable.html)
* [Apache\* MXNet](https://mxnet.apache.org), including [1.6.0](https://github.com/apache/incubator-mxnet/tree/1.6.0), [1.7.0](https://github.com/apache/incubator-mxnet/tree/1.7.0), [1.8.0](https://github.com/apache/incubator-mxnet/tree/1.8.0)
* [ONNX\* Runtime](https://github.com/microsoft/onnxruntime), including [1.6.0](https://github.com/microsoft/onnxruntime/tree/v1.6.0), [1.7.0](https://github.com/microsoft/onnxruntime/tree/v1.7.0), [1.8.0](https://github.com/microsoft/onnxruntime/tree/v1.8.0)


## Installation

Select the installation based on your operating system.


### Linux Installation

You can install LPOT using one of three options: Install just the LPOT library
from binary or source, or get the Intel-optimized framework together with the
LPOT library by installing the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

#### Option 1 Install from binary

  ```Shell
  # install stable version from pip
  pip install lpot

  # install nightly version from pip
  pip install -i https://test.pypi.org/simple/ lpot

  # install stable version from from conda
  conda install lpot -c conda-forge -c intel 
  ```

#### Option 2 Install from source

  ```Shell
  git clone https://github.com/intel/lpot.git
  cd lpot
  pip install -r requirements.txt
  python setup.py install
  ```

#### Option 3 Install from AI Kit

The Intel® LPOT library is released as part of the
[Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).
The AI Kit provides a consolidated package of Intel's latest deep learning and
machine optimizations all in one place for ease of development. Along with
LPOT, the AI Kit includes Intel-optimized versions of deep learning frameworks
(such as TensorFlow and PyTorch) and high-performing Python libraries to
streamline end-to-end data science and AI workflows on Intel architectures.

The AI Kit is distributed through many common channels,
including from Intel's website, YUM, APT, Anaconda, and more.
Select and [download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/download.html)
the AI Kit distribution package that's best suited for you and follow the
[Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html)
for post-installation instructions.

|[Download AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/) |[AI Kit Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) |
|---|---|

### Windows Installation

**Prerequisites**

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.6 or 3.7 or 3.8 or 3.9

- Download and install [anaconda](https://anaconda.org/).

- Create a virtual environment named lpot in anaconda:

    ```shell
    # Here we install python 3.7 for instance. You can also choose python 3.6, 3.8, or 3.9.
    conda create -n lpot python=3.7
    conda activate lpot
    ```
**Installation options**

#### Option 1 Install from binary

  ```Shell
  # install stable version from pip
  pip install lpot

  # install nightly version from pip
  pip install -i https://test.pypi.org/simple/ lpot

  # install from conda
  conda install lpot -c conda-forge -c intel 
  ```

#### Option 2 Install from source

```shell
git clone https://github.com/intel/lpot.git
cd lpot
pip install -r requirements.txt
python setup.py install
```

## Documentation

**Get Started**

* [APIs](docs/api-introduction.md) explains Intel® Low Precision Optimization Tool's API.
* [Transform](docs/transform.md) introduces how to utilize LPOT's built-in data processing and how to develop a custom data processing method. 
* [Dataset](docs/dataset.md) introduces how to utilize LPOT's built-in dataset and how to develop a custom dataset.
* [Metric](docs/metric.md) introduces how to utilize LPOT's built-in metrics and how to develop a custom metric.
* [Tutorial](docs/tutorial.md) provides comprehensive instructions on how to utilize LPOT's features with examples. 
* [Examples](/examples) are provided to demonstrate the usage of LPOT in different frameworks: TensorFlow, PyTorch, MXNet, and ONNX Runtime.
* [UX](docs/ux.md) is a web-based system used to simplify LPOT usage.
* [Intel oneAPI AI Analytics Toolkit Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) explains the AI Kit components, installation and configuration guides, and instructions for building and running sample apps.
* [AI and Analytics Samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics) includes code samples for Intel oneAPI libraries.

**Deep Dive**

* [Quantization](docs/Quantization.md) are processes that enable inference and training by performing computations at low-precision data types, such as fixed-point integers. LPOT supports Post-Training Quantization ([PTQ](docs/PTQ.md)) with [different quantization capabilities](docs/backend_quant.md) and Quantization-Aware Training ([QAT](docs/QAT.md)). Note that ([Dynamic Quantization](docs/dynamic_quantization.md)) currently has limited support.
* [Pruning](docs/pruning.md) provides a common method for introducing sparsity in weights and activations.
* [Benchmarking](docs/benchmark.md) introduces how to utilize the benchmark interface of LPOT.
* [Mixed precision](docs/mixed_precision.md) introduces how to enable mixed precision, including BFP16 and int8 and FP32, on Intel platforms during tuning.
* [Graph Optimization](docs/graph_optimization.md) introduces how to enable graph optimization for FP32 and auto-mixed precision.
* [Model Conversion](docs/model_conversion.md) introduces how to convert TensorFlow QAT model to quantized model running on Intel platforms.
* [TensorBoard](docs/tensorboard.md) provides tensor histograms and execution graphs for tuning debugging purposes. 

**Advanced Topics**

* [Adaptor](docs/adaptor.md) is the interface between LPOT and framework. The method to develop adaptor extension is introduced with ONNX Runtime as example. 
* [Strategy](docs/tuning_strategies.md) can automatically optimized low-precision recipes for deep learning models to achieve optimal product objectives like inference performance and memory usage with expected accuracy criteria. The method to develop a new strategy is introduced.

**Publications**

* [MLPerf™ Performance Gains Abound with latest 3rd Generation Intel® Xeon® Scalable Processors](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/3rd-gen-xeon-mlperf-performance-gains.html) (Apr 2021)
* [3D Digital Face Reconstruction Solution enabled by 3rd Gen Intel® Xeon® Scalable Processors](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/tencent-3d-digital-face-reconstruction.html) (Apr 2021)
* [Accelerating Alibaba Transformer model performance with 3rd Gen Intel® Xeon® Scalable Processors (Ice Lake) and Intel® Deep Learning Boost](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/alibaba-lpot.html) (Apr 2021)
* [Using Low-Precision Optimizations for High-Performance DL Inference Applications](https://techdecoded.intel.io/essentials/using-low-precision-optimizations-for-high-performance-dl-inference-applications/#gs.z20k91) (Apr 2021)
* [DL Boost Quantization with CERN's 3D-GANs model](https://www.nextplatform.com/2021/02/01/cern-uses-dlboost-oneapi-to-juice-inference-without-accuracy-loss/) (Feb 2021)

Full publication list please refers to [here](docs/publication_list.md)

## System Requirements

Intel® Low Precision Optimization Tool supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64), specially optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper Lake, and Icelake)
* future Intel Xeon Scalable processor (code name Sapphire Rapids)

Intel® Low Precision Optimization Tool requires installing the pertinent Intel-optimized framework version for TensorFlow, PyTorch, MXNet, and ONNX runtime.

### Validated Hardware/Software Environment

<table class="docutils">
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
    <td class="tg-nrix" rowspan="20">Cascade Lake<br><br>Cooper Lake<br><br>Skylake<br><br>Ice Lake</td>
    <td class="tg-nrix" rowspan="20">CentOS 8.3<br><br>Ubuntu 18.04</td>
    <td class="tg-nrix" rowspan="20">3.6<br><br>3.7<br><br>3.8<br><br>3.9</td>
    <td class="tg-cly1" rowspan="10">TensorFlow</td>
    <td class="tg-7zrl">2.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.5.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.4.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.3.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.2.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.1.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP1</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP2</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP3</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.2</td>
  </tr>
  <tr>
    <td class="tg-7zrl" rowspan="4">PyTorch</td>
    <td class="tg-7zrl">1.5.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.6.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.8.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">IPEX</td>
  </tr>
  <tr>
    <td class="tg-cly1" rowspan="2">MXNet</td>
    <td class="tg-7zrl">1.8.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.7.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl" rowspan="3">ONNX Runtime</td>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.7.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.8.0</td>
  </tr>
</tbody>
</table>

### Validated Models

Intel® Low Precision Optimization Tool provides numerous examples to show promising accuracy loss with the best performance gain. A full quantized model list on various frameworks is available in the [Model List](docs/full_model_list.md).

#### Validated MLPerf Models

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Framework</th>
    <th>Support</th>
    <th>Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">ResNet50 v1.5</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/tensorflow/image_recognition">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/pytorch/ipex/image_recognition/imagenet/cpu/ptq">Link</a></td>
  </tr>
  <tr>
    <td>DLRM</td>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/pytorch/fx/recommendation">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2">BERT-large</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/tensorflow/nlp/bert_large_squad">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/pytorch/eager/language_translation/ptq">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2">SSD-ResNet34</td>
    <td>TensorFlow</td>
    <td>WIP</td>
    <td></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/pytorch/fx/object_detection/ssd_resnet34/ptq">Link</a></td>
  </tr>
  <tr>
    <td>RNN-T</td>
    <td>PyTorch</td>
    <td>WIP</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">3D-UNet</td>
    <td>TensorFlow</td>
    <td>WIP</td>
    <td></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="https://github.com/intel/lpot/tree/master/examples/pytorch/eager/medical_imaging/3d-unet">Link</a></td>
  </tr>
</tbody>
</table>

#### Validated Quantized Models

<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th>Performance speed up</th>
  </tr>
  <tr>
    <td>INT8 Tuning Accuracy</td>
    <td>FP32 Accuracy Baseline</td>
    <td>Acc Ratio [(INT8-FP32)/FP32]</td>
    <td>Realtime Latency Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>resnet50v1.0</td>
    <td>74.24%</td>
    <td>74.27%</td>
    <td>-0.04%</td>
    <td>2.67x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>resnet50v1.5</td>
    <td>76.94%</td>
    <td>76.46%</td>
    <td>0.63%</td>
    <td>2.54x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>resnet101</td>
    <td>77.21%</td>
    <td>76.45%</td>
    <td>0.99%</td>
    <td>2.46x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>inception_v1</td>
    <td>70.30%</td>
    <td>69.74%</td>
    <td>0.80%</td>
    <td>1.63x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>inception_v2</td>
    <td>74.27%</td>
    <td>73.97%</td>
    <td>0.41%</td>
    <td>1.74x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>inception_v3</td>
    <td>77.29%</td>
    <td>76.75%</td>
    <td>0.70%</td>
    <td>2.11x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>inception_v4</td>
    <td>80.36%</td>
    <td>80.27%</td>
    <td>0.11%</td>
    <td>2.59x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>inception_resnet_v2</td>
    <td>80.42%</td>
    <td>80.40%</td>
    <td>0.02%</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>mobilenetv1</td>
    <td>73.93%</td>
    <td>70.96%</td>
    <td>4.19%</td>
    <td>2.27x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>mobilenetv2</td>
    <td>71.96%</td>
    <td>71.76%</td>
    <td>0.28%</td>
    <td>1.78x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>vgg16</td>
    <td>72.13%</td>
    <td>70.89%</td>
    <td>1.75%</td>
    <td>3.86x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.5.0</td>
    <td>vgg19</td>
    <td>72.35%</td>
    <td>71.01%</td>
    <td>1.89%</td>
    <td>3.90x</td>
  </tr>
</tbody>
</table>


<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th>Performance speed up</th>
  </tr>
  <tr>
    <td>INT8 Tuning Accuracy</td>
    <td>FP32 Accuracy Baseline</td>
    <td>Acc Ratio [(INT8-FP32)/FP32]</td>
    <td>Realtime Latency Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>resnet18</td>
    <td>69.58%</td>
    <td>69.76%</td>
    <td>-0.26%</td>
    <td>2.13x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>resnet50</td>
    <td>75.87%</td>
    <td>76.13%</td>
    <td>-0.34%</td>
    <td>3.11x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>resnext101_32x8d</td>
    <td>79.09%</td>
    <td>79.31%</td>
    <td>-0.28%</td>
    <td>4.99x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_base_mrpc</td>
    <td>87.92%</td>
    <td>88.73%</td>
    <td>-0.91%</td>
    <td>1.79x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_base_cola</td>
    <td>58.33%</td>
    <td>58.84%</td>
    <td>-0.87%</td>
    <td>2.01x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_base_sts-b</td>
    <td>88.46%</td>
    <td>89.27%</td>
    <td>-0.91%</td>
    <td>2.01x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_base_sst-2</td>
    <td>91.97%</td>
    <td>91.86%</td>
    <td>0.12%</td>
    <td>2.00x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_base_rte</td>
    <td>69.68%</td>
    <td>69.68%</td>
    <td>0.00%</td>
    <td>1.92x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_large_mrpc</td>
    <td>87.60%</td>
    <td>88.33%</td>
    <td>-0.83%</td>
    <td>2.47x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_large_squad</td>
    <td>92.99</td>
    <td>93.05</td>
    <td>-0.06%</td>
    <td>1.85x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_large_qnli</td>
    <td>91.12%</td>
    <td>91.82%</td>
    <td>-0.76%</td>
    <td>2.47x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_large_rte</td>
    <td>71.84%</td>
    <td>72.56%</td>
    <td>-0.99%</td>
    <td>1.44x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.8.0+cpu</td>
    <td>bert_large_cola</td>
    <td>61.97%</td>
    <td>62.57%</td>
    <td>-0.97%</td>
    <td>2.48x</td>
  </tr>
</tbody>
</table>

#### Validated Pruning Models

<table>
<thead>
  <tr>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">FWK</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">fp32 baseline</th>
    <th colspan="3">gradient sensitivity with 20% sparsity</th>
    <th colspan="3">+onnx dynamic quantization on pruned model</th>
  </tr>
  <tr>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>perf gain (sample/s)</td>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>perf gain (sample/s)</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SST-2</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>accuracy = 92.32</td>
    <td>accuracy = 91.97</td>
    <td>-0.38</td>
    <td>1.30x</td>
    <td>accuracy = 92.20</td>
    <td>-0.13</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[accuracy, f1] = [91.10, 88.05]</td>
    <td>[accuracy, f1] = [89.97, 86.54]</td>
    <td>[-1.24, -1.71]</td>
    <td>1.32x</td>
    <td>[accuracy, f1] = [89.75, 86.60]</td>
    <td>[-1.48, -1.65]</td>
    <td>1.81x</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">FWK</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">fp32 baseline</th>
    <th colspan="2">Pattern Lock on 70% Unstructured Sparsity</th>
    <th colspan="2">Pattern Lock on 50% 1:2 Structured Sparsity</th>
  </tr>
  <tr>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>accuracy%</td>
    <td> drop%</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MNLI</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[m, mm] = [84.57, 84.79]</td>
    <td>[m, mm] = [82.45, 83.27]</td>
    <td>[-2.51, -1.80]</td>
    <td>[m, mm] = [83.20, 84.11]</td>
    <td>[-1.62, -0.80]</td>
  </tr>
  <tr>
    <td>SST-2</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>accuracy = 92.32</td>
    <td>accuracy = 91.51</td>
    <td>-0.88</td>
    <td>accuracy = 92.20</td>
    <td>-0.13</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[accuracy, f1] = [91.10, 88.05]</td>
    <td>[accuracy, f1] = [90.48, 87.06]</td>
    <td>[-0.68, -1.12]</td>
    <td>[accuracy, f1] = [90.92, 87.78]</td>
    <td>[-0.20, -0.31]</td>
  </tr>
  <tr>
    <td>QNLI</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>accuracy = 91.54</td>
    <td>accuracy = 90.39</td>
    <td>-1.26</td>
    <td>accuracy = 90.87</td>
    <td>-0.73</td>
  </tr>
  <tr>
    <td>QnA</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[em, f1] = [79.34, 87.10]</td>
    <td>[em, f1] = [77.27, 85.75]</td>
    <td>[-2.61, -1.54]</td>
    <td>[em, f1] = [78.03, 86.50]</td>
    <td>[-1.65, -0.69]</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th>Framework</th>
    <th>Model</th>
    <th>fp32 baseline</th>
    <th>Compression</th>
    <th>dataset</th>
    <th>acc(drop)%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Pytorch</td>
    <td>resnet18</td>
    <td>69.76</td>
    <td>30% sparsity on magnitude</td>
    <td>ImageNet</td>
    <td>69.47(-0.42)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet18</td>
    <td>69.76</td>
    <td>30% sparsity on gradient sensitivity</td>
    <td>ImageNet</td>
    <td>68.85(-1.30)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude</td>
    <td>ImageNet</td>
    <td>76.11(-0.03)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude and post training quantization</td>
    <td>ImageNet</td>
    <td>76.01(-0.16)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude and quantization aware training</td>
    <td>ImageNet</td>
    <td>75.90(-0.30)</td>
  </tr>
</tbody>
</table>

## Additional Content

* [Release Information](releases_info.md)
* [Contribution Guidelines](contributions.md)
* [Legal](legal_information.md)
* [Security Policy](security_policy.md)
* [Intel® LPOT Website](https://intel.github.io/lpot)

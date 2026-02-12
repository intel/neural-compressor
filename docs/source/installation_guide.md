# Installation

1. [Installation](#installation)

    1.1. [Prerequisites](#prerequisites)

    1.2. [Install from Binary](#install-from-binary)

    1.3. [Install from Source](#install-from-source)

    1.4. [Install from AI Kit](#install-from-ai-kit)

2. [System Requirements](#system-requirements)

   2.1. [Validated Hardware Environment](#validated-hardware-environment)

   2.2. [Validated Software Environment](#validated-software-environment)

## Installation
### Prerequisites
You can install Neural Compressor using one of three options: Install single component from binary or source, or get the Intel-optimized framework together with the library by installing the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.10 or 3.11 or 3.12

> Notes:
> - If you get some build issues, please check [frequently asked questions](faq.md) at first.

### Install Framework
#### Install torch for CPU
```Shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
#### Use Docker Image with torch installed for HPU
https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click 

#### Install torch/intel_extension_for_pytorch for Intel GPU
https://intel.github.io/intel-extension-for-pytorch/index.html#installation 

#### Install torch for other platform
https://pytorch.org/get-started/locally

#### Install tensorflow
```Shell
pip install tensorflow
```

### Install from Binary
- Install from Pypi
```Shell
# Install 2.X API + Framework extension API + PyTorch dependency
pip install neural-compressor[pt]
```
```Shell
# Install 2.X API + Framework extension API + TensorFlow dependency
pip install neural-compressor[tf]
```
```Shell
# Install 2.X API + Framework extension API
# With this install CMD, some dependencies for framework extension API not installed, 
# you can install them separately by `pip install -r requirements_pt.txt` or `pip install -r requirements_tf.txt`.
pip install neural-compressor
```
```Shell
# Framework extension API + PyTorch dependency
pip install neural-compressor-pt
```
```Shell
# Framework extension API + TensorFlow dependency
pip install neural-compressor-tf
```

### Install from Source

```Shell
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -r requirements.txt
```

```Shell
# PyTorch framework extension API + PyTorch dependency
INC_PT_ONLY=1 pip install .
```

```Shell
# TensorFlow framework extension API + TensorFlow dependency
INC_TF_ONLY=1 pip install .
```

```Shell
# JAX framework extension API + JAX dependency
INC_JAX_ONLY=1 pip install .
```

```Shell
# Full Installation
pip install .
[optional] pip install -r requirements_pt.txt # for PyTorch framework extension API
[optional] pip install -r requirements_tf.txt # for TensorFlow framework extension API
```

### Install from AI Kit

The Intel® Neural Compressor library is released as part of the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit). The AI Kit provides a consolidated package of Intel's latest deep learning and machine optimizations all in one place for ease of development. Along with Neural Compressor, the AI Kit includes Intel-optimized versions of deep learning frameworks (such as TensorFlow and PyTorch) and high-performing Python libraries to streamline end-to-end data science and AI workflows on Intel architectures.

The AI Kit is distributed through many common channels, including from Intel's website, YUM, APT, Anaconda, and more. Select and [download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/download.html) the AI Kit distribution package that's best suited for you and follow the [Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) for post-installation instructions.

|Download|Guide|
|-|-|
|[Download AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/) |[AI Kit Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) |

## System Requirements

### Validated Hardware Environment

#### Intel® Neural Compressor supports HPUs based on heterogeneous architecture with two compute engines (MME and TPC): 
* Intel Gaudi Al Accelerators (Gaudi2, Gaudi3)

#### Intel® Neural Compressor supports CPUs based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):

* Intel Xeon Scalable processor (Sapphire Rapids, Emerald Rapids, Granite Rapids)
* Intel Xeon CPU Max Series (Sapphire Rapids HBM)
* Intel Core Ultra Processors (Meteor Lake, Lunar Lake)

#### Intel® Neural Compressor supports GPUs built on Intel's Xe architecture:

* Intel Data Center GPU Flex Series (Arctic Sound-M)
* Intel Data Center GPU Max Series (Ponte Vecchio)
* Intel® Arc™ B-Series Graphics (Battlemage)

#### Intel® Neural Compressor quantized ONNX models support multiple hardware vendors through ONNX Runtime:

* Intel CPU, AMD/ARM CPU, and NVidia GPU. Please refer to the validated model [list](./validated_model_list.md#validated-onnx-qdq-int8-models-on-multiple-hardware-through-onnx-runtime).

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 24.04, MacOS Ventura 13.5, Windows 11
* Python version: 3.10, 3.11, 3.12

<table class="docutils">
<thead>
  <tr style="vertical-align: middle; text-align: center;">
    <th>Framework</th>
    <th>TensorFlow</th>
    <th>Intel®<br>Extension for<br>TensorFlow*</th>
    <th>PyTorch</th>
    <th>Intel®<br>Extension for<br>PyTorch*</th>
    <th>ONNX<br>Runtime</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td class="tg-7zrl">
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.16.1>2.16.1</a><br>
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.15.0>2.15.0</a><br>
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.14.1>2.14.1</a><br></td>
    <td class="tg-7zrl"> 
    <a href=https://github.com/intel/intel-extension-for-tensorflow/tree/v2.15.0.0>2.15.0.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-tensorflow/tree/v2.14.0.1>2.14.0.1</a><br>
    <a href=https://github.com/intel/intel-extension-for-tensorflow/tree/v2.13.0.0>2.13.0.0</a><br></td>
    <td class="tg-7zrl">
    <a href=https://github.com/pytorch/pytorch/tree/v2.8.0>2.8.0</a><br>
    <a href=https://github.com/pytorch/pytorch/tree/v2.7.1>2.7.1</a><br>
    <a href=https://github.com/pytorch/pytorch/tree/v2.6.0>2.6.0</a><br></td>
    <td class="tg-7zrl">
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v2.8.0%2Bcpu>2.8.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v2.7.0%2Bcpu>2.7.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v2.6.0%2Bcpu>2.6.0</a><br></td>
    <td class="tg-7zrl">
    <a href=https://github.com/microsoft/onnxruntime/tree/v1.18.0>1.18.0</a><br>
    <a href=https://github.com/microsoft/onnxruntime/tree/v1.17.3>1.17.3</a><br>
    <a href=https://github.com/microsoft/onnxruntime/tree/v1.16.3>1.16.3</a><br></td>
  </tr>
</tbody>
</table>

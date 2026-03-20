# Installation

1. [Installation](#installation)

    1.1. [Prerequisites](#prerequisites)

    1.2. [Install from Binary](#install-from-binary)

    1.3. [Install from Source](#install-from-source)

2. [System Requirements](#system-requirements)

   2.1. [Validated Hardware Environment](#validated-hardware-environment)

   2.2. [Validated Software Environment](#validated-software-environment)

## Installation
### Prerequisites

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.10 or 3.11 or 3.12 or 3.13

> Notes:
> - If you get some build issues, please check [frequently asked questions](faq.md) at first.

### Install Framework for PyTorch Backend (on-demand)
Intel Neural Compressor supports PyTorch with CPU, GPU and HPU. Please install the corresponding PyTorch version based on your hardware environment.
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

### Install from Binary
- Install from Pypi
```Shell
# Framework extension API for PyTorch/Tensorflow/JAX
pip install neural-compressor
# Framework extension API + corresponding framework dependency
pip install neural-compressor[pt]
pip install neural-compressor[tf]
pip install neural-compressor[jax] # JAX support is available since v3.8
```
```Shell
# Framework extension API + PyTorch dependency
pip install neural-compressor-pt
```
```Shell
# Framework extension API + TensorFlow dependency
pip install neural-compressor-tf
```
```Shell
# Framework extension API + JAX dependency, available since v3.8
pip install neural-compressor-jax
```

### Install from Source
The latest code on master branch may not be stable. Please switch to the latest release tag for better stability. Feel free to open an [issue](https://github.com/intel/neural-compressor/issues) if you encounter an error.  
```Shell
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
git fetch --tags && git checkout "$(git tag -l 'v*' --sort=-v:refname | head -n 1)"
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
# JAX framework extension API + JAX dependency, available since v3.8
INC_JAX_ONLY=1 pip install .
```

## System Requirements

### Validated Hardware Environment

#### Intel® Neural Compressor supports HPUs based on heterogeneous architecture with two compute engines (MME and TPC): 
* Intel Gaudi Al Accelerators (Gaudi2, Gaudi3)

#### Intel® Neural Compressor supports CPUs based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):

* Intel Xeon Scalable processor (Sapphire Rapids, Emerald Rapids, Granite Rapids)
* Intel Xeon CPU Max Series (Sapphire Rapids HBM)

#### Intel® Neural Compressor supports GPUs built on Intel's Xe architecture:

* Intel® Arc™ B-Series Graphics (Battlemage)

### Validated Software Environment

* OS version: Ubuntu 24.04, MacOS Ventura 13.5, Windows 11
* Python version: 3.11, 3.12, 3.13

<table class="docutils">
<thead>
  <tr style="vertical-align: middle; text-align: center;">
    <th>Framework</th>
    <th>TensorFlow</th>
    <th>PyTorch</th>
    <th>JAX</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td class="tg-7zrl">
    <a href=https://github.com/tensorflow/tensorflow/releases/tag/v2.19.0>2.19.0</a><br></td>
    <td class="tg-7zrl">
    <a href=https://github.com/pytorch/pytorch/releases/tag/v2.10.0>2.10.0</a><br>
    <a href=https://github.com/pytorch/pytorch/releases/tag/v2.9.1>2.9.1</a><br></td>
    <td class="tg-7zrl">
    <a href=https://github.com/jax-ml/jax/releases/tag/jax-v0.9.1>0.9</a><br></td>
  </tr>
</tbody>
</table>

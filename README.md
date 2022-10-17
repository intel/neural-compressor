<div align="center">
  
Intel® Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime, and MXNet)</h3>

[![python](https://img.shields.io/badge/python-3.7%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-1.14-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/master/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-90%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)
</div>

---
<div align="left">

Intel® Neural Compressor, formerly known as Intel® Low Precision Optimization Tool, is an open-source Python library that runs on Intel CPUs and GPUs, which delivers unified interfaces across multiple deep-learning frameworks for popular network compression technologies such as quantization, pruning, and knowledge distillation. This tool supports automatic accuracy-driven tuning strategies to help the user quickly find out the best quantized model. It also implements different weight-pruning algorithms to generate a pruned model with predefined sparsity goal. It also supports knowledge distillation to distill the knowledge from the teacher model to the student model. 
Intel® Neural Compressor is a critical AI software component in the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

> **Note:**
> GPU support is under development.

**Visit the Intel® Neural Compressor online document website at: <https://intel.github.io/neural-compressor>.**   

## Installation

### Prerequisites

Python version: 3.7, 3.8, 3.9, 3.10

### Install on Linux
- Release binary install 
  ```Shell
  # install stable basic version from pip
  pip install neural-compressor
  # Or install stable full version from pip (including GUI)
  pip install neural-compressor-full
  ```
- Nightly binary install
  ```Shell
  git clone https://github.com/intel/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  # install nightly basic version from pip
  pip install -i https://test.pypi.org/simple/ neural-compressor
  # Or install nightly full version from pip (including GUI)
  pip install -i https://test.pypi.org/simple/ neural-compressor-full
  ```
More installation methods can be found at [Installation Guide](./docs/installation_guide.md). Please check out our [FAQ](./docs/faq.md) for more details.

## Getting Started
### Quantization with Python API    

```shell
# A TensorFlow Example
pip install tensorflow
# Prepare fp32 model
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb
```
```python
import tensorflow as tf
from neural_compressor.experimental import Quantization, common
tf.compat.v1.disable_eager_execution()
quantizer = Quantization()
quantizer.model = './mobilenet_v1_1.0_224_frozen.pb'
dataset = quantizer.dataset('dummy', shape=(1, 224, 224, 3))
quantizer.calib_dataloader = common.DataLoader(dataset)
quantizer.fit()
```
### Quantization with [GUI](./docs/bench.md)
```shell
# An ONNX Example
pip install onnx==1.12.0 onnxruntime==1.12.1 onnxruntime-extensions
# Prepare fp32 model
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-12.onnx
# Start GUI
inc_bench
```
<a target="_blank" href="./docs/imgs/INC_GUI.gif">
  <img src="./docs/imgs/INC_GUI.gif" alt="Architecture">
</a>

### Quantization with [Auto-coding API](./neural_coder/docs/AutoQuant.md) (Experimental)

```python
from neural_coder import auto_quant
auto_quant(
    code="https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py",
    args="--model_name_or_path albert-base-v2 \
          --task_name sst2 \
          --do_eval \
          --output_dir result \
          --overwrite_output_dir",
)
```

## System Requirements

Intel® Neural Compressor supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64) that are specifically optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper Lake, and Icelake)
* Future Intel Xeon Scalable processor (code name Sapphire Rapids)

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 20.04  
* Python version: 3.7, 3.8, 3.9, 3.10  

<table class="docutils">
<thead>
  <tr>
    <th>Framework</th>
    <th>TensorFlow</th>
    <th>Intel TensorFlow</th>
    <th>PyTorch</th>
    <th>IPEX</th>
    <th>ONNX Runtime</th>
    <th>MXNet</th>
  </tr>
</thead>
<tbody>
  <tr align="center">
    <th>Version</th>
    <td class="tg-7zrl"><a href=https://github.com/tensorflow/tensorflow/tree/v2.10.0>2.10.0</a><br>
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.9.1>2.9.1</a><br>
    <a href=https://github.com/tensorflow/tensorflow/tree/v2.8.2>2.8.2</a><br>
    <td class="tg-7zrl"><a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.10.0>2.10.0</a><br>
    <a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.9.1>2.9.1</a><br>
    <a href=https://github.com/Intel-tensorflow/tensorflow/tree/v2.8.0>2.8.0</a><br>
    <td class="tg-7zrl"><a href=https://download.pytorch.org/whl/torch_stable.html>1.12.1+cpu</a><br>
    <a href=https://download.pytorch.org/whl/torch_stable.html>1.11.0+cpu</a><br>
    <a href=https://download.pytorch.org/whl/torch_stable.html>1.10.0+cpu</a></td>
    <td class="tg-7zrl"><a href=https://github.com/intel/intel-extension-for-pytorch/tree/v1.12.0>1.12.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/1.11.0>1.11.0</a><br>
    <a href=https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.0>1.10.0</a></td>
    <td class="tg-7zrl"><a href=https://github.com/microsoft/onnxruntime/tree/v1.12.1>1.12.1</a><br>
    <a href=https://github.com/microsoft/onnxruntime/tree/v1.11.0>1.11.0</a><br>
    <a href=https://github.com/microsoft/onnxruntime/tree/v1.10.0>1.10.0</a></td>
    <td class="tg-7zrl"><a href=https://github.com/apache/incubator-mxnet/tree/1.8.0>1.8.0</a><br>
    <a href=https://github.com/apache/incubator-mxnet/tree/1.7.0>1.7.0</a><br>
    <a href=https://github.com/apache/incubator-mxnet/tree/1.6.0>1.6.0</a></td>
  </tr>
</tbody>
</table>

> **Note:**
> Set the environment variable ``TF_ENABLE_ONEDNN_OPTS=1`` to enable oneDNN optimizations if you are using TensorFlow v2.6 to v2.8. oneDNN is the default for TensorFlow v2.9.

### Validated Models
Intel® Neural Compressor validated 420+ [examples](./examples) for quantization with a performance speedup geomean of 2.2x and up to 4.2x on VNNI while minimizing accuracy loss. Over 30 pruning and knowledge distillation samples are also available. More details for validated models are available [here](docs/validated_model_list.md).   

<div style = "width: 77%; margin-bottom: 2%;">
  <a target="_blank" href="./docs/imgs/release_data.png">
    <img src="./docs/imgs/release_data.png" alt="Architecture" width=800 height=500>
  </a>
</div>

## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="9">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3" align="center"><a href="docs/design.md">Architecture</a></td>
      <td colspan="2" align="center"><a href="https://github.com/intel/neural-compressor/tree/master/examples">Examples</a></td>
      <td colspan="2" align="center"><a href="docs/bench.md">GUI</a></td>
      <td colspan="2" align="center"><a href="docs/api-introduction.md">APIs</a></td>
    </tr>
    <tr>
      <td colspan="5" align="center"><a href="https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html">Intel oneAPI AI Analytics Toolkit</a></td>
      <td colspan="4" align="center"><a href="https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics">AI and Analytics Samples</a></td>
    </tr>
  </tbody>
  <thead>
  <tr>
    <th colspan="9">Basic API</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2" align="center"><a href="docs/transform.md">Transform</a></td>
      <td colspan="2" align="center"><a href="docs/dataset.md">Dataset</a></td>
      <td colspan="2" align="center"><a href="docs/metric.md">Metric</a></td>
      <td colspan="3" align="center"><a href="docs/objective.md">Objective</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="9">Deep Dive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="2" align="center"><a href="docs/Quantization.md">Quantization</a></td>
        <td colspan="1" align="center"><a href="docs/pruning.md">Pruning</a> <a href="docs/sparsity.md">(Sparsity)</a> </td> 
        <td colspan="2" align="center"><a href="docs/distillation.md">Knowledge Distillation</a></td>
        <td colspan="2" align="center"><a href="docs/mixed_precision.md">Mixed Precision</a></td>
        <td colspan="2" align="center"><a href="docs/orchestration.md">Orchestration</a></td>
    </tr>
    <tr>
        <td colspan="2" align="center"><a href="docs/benchmark.md">Benchmarking</a></td>
        <td colspan="3" align="center"><a href="docs/distributed.md">Distributed Training</a></td>
        <td colspan="2" align="center"><a href="docs/model_conversion.md">Model Conversion</a></td>
        <td colspan="2" align="center"><a href="docs/tensorboard.md">TensorBoard</a></td>
    </tr>
    <tr>
        <td colspan="4" align="center"><a href="docs/distillation_quantization.md">Distillation for Quantization</a></td>
        <td colspan="5" align="center"><a href="neural_coder">Neural Coder (No-Code Solution)</a></td>
    </tr>    
    
  </tbody>
  <thead>
      <tr>
        <th colspan="9">Advanced Topics</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="3" align="center"><a href="docs/adaptor.md">Adaptor</a></td>
          <td colspan="3" align="center"><a href="docs/tuning_strategies.md">Strategy</a></td>
          <td colspan="3" align="center"><a href="docs/reference_examples.md">Reference Example</a></td>
      </tr>
  </tbody>
</table>

## Selected Publications

* [Neural Coder (Intel Neural Compressor Plug-in): One-Click, No-Code Solution (Pat's Keynote IntelON 2022)](https://twitter.com/i/status/1574909338203967497) (Sep 2022)
* [Alibaba Cloud and Intel Neural Compressor Deliver Better Productivity for PyTorch Users](https://medium.com/intel-analytics-software/alibaba-cloud-collaborates-with-intel-neural-compressor-for-better-productivity-and-performance-83cdb6500420) (Sep 2022)
* [Efficient Text Classification with Intel Neural Compressor](https://medium.com/intel-analytics-software/efficient-text-classification-with-intel-neural-compressor-4853296deeac) (Sep 2022)
* [Dynamic Neural Architecture Search with Intel Neural Compressor](https://medium.com/intel-analytics-software/dynamic-neural-architecture-search-with-intel-neural-compressor-7b05eaf325f3) (Sep 2022)
* [Easy Quantization in PyTorch Using Fine-Grained FX](https://medium.com/intel-analytics-software/easy-quantization-in-pytorch-using-fine-grained-fx-80be2c4bc2d6) (Sep 2022)
* [One-Click Enabling of Intel Neural Compressor Features in PyTorch Scripts](https://medium.com/intel-analytics-software/one-click-enable-intel-neural-compressor-features-in-pytorch-scripts-5d4e31f5a22b) (Aug 2022)
* [Deep learning inference optimization for Address Purification](https://zhuanlan.zhihu.com/p/552484413?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=667097517833981952) (Aug 2022)

> View our [full publication list](docs/publication_list.md).

## Additional Content

* [Release Information](docs/releases_info.md)
* [Contribution Guidelines](docs/contributions.md)
* [Legal Information](docs/legal_information.md)
* [Security Policy](docs/security_policy.md)
* [Intel® Neural Compressor Website](https://intel.github.io/neural-compressor)

## Hiring

We are actively hiring. Send your resume to inc.maintainers@intel.com if you are interested in model compression techniques.

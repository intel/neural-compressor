<div align="center">

Intel® Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, and ONNX Runtime)</h3>

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-3.7-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/master/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-85%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)

[Architecture](./docs/source/design.md#architecture)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Workflow](./docs/source/design.md#workflows)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[LLMs Recipes](./docs/source/llm_recipes.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentations](https://intel.github.io/neural-compressor)

---
<div align="left">

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and [ONNX Runtime](https://onnxruntime.ai/),
as well as Intel extensions such as [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).
In particular, the tool provides the key features, typical examples, and open collaborations as below:

* Support a wide range of Intel hardware such as [Intel Gaudi Al Accelerators](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html), [Intel Core Ultra Processors](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html), [Intel Xeon Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html), [Intel Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/overview.html), and [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/overview.html) with extensive testing;
support AMD CPU, ARM CPU, and NVidia GPU through ONNX Runtime with limited testing; support NVidia GPU for some WOQ algorithms like AutoRound and HQQ.

* Collaborate with cloud marketplaces such as [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/bitnami-launchpad/inc-tensorflow-intel?project=verdant-sensor-286207), [Amazon Web Services](https://aws.amazon.com/marketplace/pp/prodview-yjyh2xmggbmga#pdp-support), and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/bitnami.inc-tensorflow-intel), software platforms such as [Tencent TACO](https://new.qq.com/rain/a/20221202A00B9S00) and [Microsoft Olive](https://github.com/microsoft/Olive), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [PyTorch](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html), [ONNX](https://github.com/onnx/models#models), [ONNX Runtime](https://github.com/microsoft/onnxruntime), and [Lightning AI](https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/advanced/post_training_quantization.rst)

## What's New
* [2025/12] [NVFP4 quantization](./docs/source/PT_NVFP4Quant.md) experimental support
* [2025/10] [MXFP8 / MXFP4 quantization](./docs/source/PT_MXQuant.md) experimental support
* [2025/09] FP8 dynamic quantization, including Linear, FusedMoE on Intel Gaudi AI Accelerators
* [2025/05] FP8 static quantization of DeepSeek V3/R1 model on Intel Gaudi AI Accelerators
* [2025/03] VLM quantization in transformers-like API on Intel CPU/GPU   

## Installation
Choose the necessary framework dependencies to install based on your deploy environment.
### Install Framework
* [Install intel_extension_for_pytorch for CPU](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)    
* [Install intel_extension_for_pytorch for Intel GPU](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)    
* [Use Docker Image with torch installed for HPU](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click)    
  **Note**: There is a version mapping between Intel Neural Compressor and Gaudi Software Stack, please refer to this [table](./docs/source/gaudi_version_map.md) and make sure to use a matched combination.    
* [Install torch for other platform](https://pytorch.org/get-started/locally)    
* [Install TensorFlow](https://www.tensorflow.org/install)    

### Install Neural Compressor from pypi
```
# Framework extension API + PyTorch dependency
pip install neural-compressor-pt
# Framework extension API + TensorFlow dependency
pip install neural-compressor-tf
```    
**Note**: Further installation methods can be found under [Installation Guide](./docs/source/installation_guide.md). check out our [FAQ](./docs/source/faq.md) for more details.

## Getting Started
After successfully installing these packages, try your first quantization program. **Following example code demonstrates FP8 Quantization**, it is supported by Intel Gaudi2 AI Accelerator.     
To try on Intel Gaudi2, docker image with Gaudi Software Stack is recommended, please refer to following script for environment setup. More details can be found in [Gaudi Guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#launch-docker-image-that-was-built).    

Run a container with an interactive shell, [more info](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#docker-installation)
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.9.0:latest
```

> Note: Since Habana software >= 1.21.0, `PT_HPU_LAZY_MODE=0` is the default setting. However, most low-precision functions (such as `convert_from_uint4`) do not support this setting. Therefore, we recommend setting `PT_HPU_LAZY_MODE=1` to maintain compatibility.

Run the example,
```python
from neural_compressor.torch.quantization import (
    FP8Config,
    prepare,
    convert,
)

import torch
import torchvision.models as models

model = models.resnet18()
qconfig = FP8Config(fp8_config="E4M3")
model = prepare(model, qconfig)

# Customer defined calibration. Below is a dummy calibration
model(torch.randn(1, 3, 224, 224).to("hpu"))

model = convert(model)

output = model(torch.randn(1, 3, 224, 224).to("hpu")).to("cpu")
print(output.shape)
```    
More [FP8 quantization doc](./docs/source/PT_FP8Quant.md).

**Following example code demonstrates weight-only large language model loading** on Intel Gaudi2 AI Accelerator. 
```python
from neural_compressor.torch.quantization import load

model_name = "TheBloke/Llama-2-7B-GPTQ"
model = load(
    model_name_or_path=model_name,
    format="huggingface",
    device="hpu",
    torch_dtype=torch.bfloat16,
)
```
**Note:** Intel Neural Compressor will convert the model format from auto-gptq to hpu format on the first load and save hpu_model.safetensors to the local cache directory for the next load. So it may take a while to load for the first time.

## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="8">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2" align="center"><a href="./docs/source/design.md#architecture">Architecture</a></td>
      <td colspan="2" align="center"><a href="./docs/source/design.md#workflows">Workflow</a></td>
      <td colspan="2" align="center"><a href="https://intel.github.io/neural-compressor/latest/docs/source/api-doc/apis.html">APIs</a></td>
      <td colspan="1" align="center"><a href="./docs/source/llm_recipes.md">LLMs Recipes</a></td>
      <td colspan="1" align="center"><a href="./examples/README.md">Examples</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="8">PyTorch Extension APIs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="8" align="center"><a href="./docs/source/PyTorch.md">Overview</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><a href="./docs/source/PT_DynamicQuant.md">Dynamic Quantization</a></td>
        <td colspan="3" align="center"><a href="./docs/source/PT_StaticQuant.md">Static Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/PT_SmoothQuant.md">Smooth Quantization</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><a href="./docs/source/PT_WeightOnlyQuant.md">Weight-Only Quantization</a></td>
        <td colspan="3" align="center"><a href="./docs/source/PT_FP8Quant.md">FP8 Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/PT_MixedPrecision.md">Mixed Precision</a></td>
    </tr>
    <tr>
        <td colspan="4" align="center"><a href="./docs/source/PT_MXQuant.md">MX Quantization</a></td>
        <td colspan="4" align="center"><a href="./docs/source/PT_NVFP4Quant.md">NVFP4 Quantization</a></td>
    </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Tensorflow Extension APIs</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="3" align="center"><a href="./docs/source/TensorFlow.md">Overview</a></td>
          <td colspan="3" align="center"><a href="./docs/source/TF_Quant.md">Static Quantization</a></td>
          <td colspan="2" align="center"><a href="./docs/source/TF_SQ.md">Smooth Quantization</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Transformers-like APIs</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/transformers_like_api.md">Overview</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Other Modules</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/autotune.md">Auto Tune</a></td>
      </tr>
  </tbody>
</table>

> **Note**:
> From 3.0 release, we recommend to use 3.X API. Compression techniques during training such as QAT, Pruning, Distillation only available in [2.X API](https://github.com/intel/neural-compressor/blob/master/docs/source/2x_user_guide.md) currently.

## Selected Publications/Events

* arXiv: [Faster Inference of LLMs using FP8 on the Intel Gaudi](https://arxiv.org/abs/2503.09975) (Mar 2025)
* PyTorch landscape: [PyTorch general optimizations](https://landscape.pytorch.org/) (Mar 2025)
* Blog on SqueezeBits: [[Intel Gaudi] #4. FP8 Quantization](https://blog.squeezebits.com/intel-gaudi-4-fp8-quantization--40269) (Jan 2025)
* EMNLP'2024: [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) (Sep 2024)
* arXiv: [Efficient Post-training Quantization with FP8 Formats](https://arxiv.org/abs/2309.14592) (Sep 2023)
* arXiv: [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) (Sep 2023)

> **Note**:
> View [Full Publication List](https://github.com/intel/neural-compressor/blob/master/docs/source/publication_list.md).

## Additional Content

* [Contribution Guidelines](./docs/source/CONTRIBUTING.md)
* [Legal Information](./docs/source/legal_information.md)
* [Security Policy](SECURITY.md)

## Communication
- [GitHub Issues](https://github.com/intel/neural-compressor/issues): mainly for bug reports, new feature requests, question asking, etc.
- [Email](mailto:inc.maintainers@intel.com): welcome to raise any interesting research ideas on model compression techniques by email for collaborations.
- [Discord Channel](https://discord.com/invite/Wxk3J3ZJkU): join the discord channel for more flexible technical discussion.
- [WeChat group](/docs/source/imgs/wechat_group.jpg): scan the QA code to join the technical discussion.

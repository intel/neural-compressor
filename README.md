<div align="center">

Intel® Neural Compressor
===========================
<h3> An open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, and ONNX Runtime)</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-3.2-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/master/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-85%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)

[Architecture](./docs/source/3x/design.md#architecture)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Workflow](./docs/source/3x/design.md#workflows)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[LLMs Recipes](./docs/source/llm_recipes.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](./docs/source/validated_model_list.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentations](https://intel.github.io/neural-compressor)

---
<div align="left">

Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and [ONNX Runtime](https://onnxruntime.ai/),
as well as Intel extensions such as [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).
In particular, the tool provides the key features, typical examples, and open collaborations as below:

* Support a wide range of Intel hardware such as [Intel Gaudi Al Accelerators](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi-overview.html), [Intel Core Ultra Processors](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html), [Intel Xeon Scalable Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html), [Intel Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html), and [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) with extensive testing;
support AMD CPU, ARM CPU, and NVidia GPU through ONNX Runtime with limited testing; support NVidia GPU for some WOQ algorithms like AutoRound and HQQ.

* Validate popular LLMs such as [LLama2](/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm), [Falcon](/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm), [GPT-J](/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm), [Bloom](/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm), [OPT](/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm), and more than 10,000 broad models such as [Stable Diffusion](/examples/pytorch/nlp/huggingface_models/text-to-image/quantization), [BERT-Large](/examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx), and [ResNet50](/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx) from popular model hubs such as [Hugging Face](https://huggingface.co/), [Torch Vision](https://pytorch.org/vision/stable/index.html), and [ONNX Model Zoo](https://github.com/onnx/models#models), with automatic [accuracy-driven](/docs/source/design.md#workflow) quantization strategies

* Collaborate with cloud marketplaces such as [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/bitnami-launchpad/inc-tensorflow-intel?project=verdant-sensor-286207), [Amazon Web Services](https://aws.amazon.com/marketplace/pp/prodview-yjyh2xmggbmga#pdp-support), and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/bitnami.inc-tensorflow-intel), software platforms such as [Alibaba Cloud](https://www.intel.com/content/www/us/en/developer/articles/technical/quantize-ai-by-oneapi-analytics-on-alibaba-cloud.html), [Tencent TACO](https://new.qq.com/rain/a/20221202A00B9S00) and [Microsoft Olive](https://github.com/microsoft/Olive), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [PyTorch](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html), [ONNX](https://github.com/onnx/models#models), [ONNX Runtime](https://github.com/microsoft/onnxruntime), and [Lightning AI](https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/advanced/post_training_quantization.rst)

## What's New
* [2024/10] [Transformers-like API](./docs/source/3x/transformers_like_api.md) for INT4 inference on Intel CPU and GPU.
* [2024/07] From 3.0 release, framework extension API is recommended to be used for quantization.
* [2024/07] Performance optimizations and usability improvements on [client-side](./docs/source/3x/client_quant.md).

## Installation
### Install Framework
#### Install torch for CPU
```Shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
#### Use Docker Image with torch installed for HPU
https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click

> **Note**:
> There is a version mapping between Intel Neural Compressor and Gaudi Software Stack, please refer to this [table](./docs/source/3x/gaudi_version_map.md) and make sure to use a matched combination.

#### Install torch/intel_extension_for_pytorch for Intel GPU
https://intel.github.io/intel-extension-for-pytorch/index.html#installation

#### Install torch for other platform
https://pytorch.org/get-started/locally

#### Install tensorflow
```Shell
pip install tensorflow
```

### Install from pypi
```Shell
# Install 2.X API + Framework extension API + PyTorch dependency
pip install neural-compressor[pt]
# Install 2.X API + Framework extension API + TensorFlow dependency
pip install neural-compressor[tf]
```
> **Note**:
> Further installation methods can be found under [Installation Guide](./docs/source/installation_guide.md). check out our [FAQ](./docs/source/faq.md) for more details.

## Getting Started

Setting up the environment:
```bash
pip install "neural-compressor>=2.3" "transformers>=4.34.0" torch torchvision
```
After successfully installing these packages, try your first quantization program.

### [FP8 Quantization](./docs/source/3x/PT_FP8Quant.md)
Following example code demonstrates FP8 Quantization, it is supported by Intel Gaudi2 AI Accelerator. 

To try on Intel Gaudi2, docker image with Gaudi Software Stack is recommended, please refer to following script for environment setup. More details can be found in [Gaudi Guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#launch-docker-image-that-was-built).
```bash
# Run a container with an interactive shell
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.19.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest
```
Run the example:
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

### Weight-Only Large Language Model Loading (LLMs)

Following example code demonstrates weight-only large language model loading on Intel Gaudi2 AI Accelerator. 

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

**Note:**

Intel Neural Compressor will convert the model format from auto-gptq to hpu format on the first load and save hpu_model.safetensors to the local cache directory for the next load. So it may take a while to load for the first time.

## Documentation

<table class="docutils">
  <thead>
  <tr>
    <th colspan="8">Overview</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2" align="center"><a href="./docs/source/3x/design.md#architecture">Architecture</a></td>
      <td colspan="2" align="center"><a href="./docs/source/3x/design.md#workflows">Workflow</a></td>
      <td colspan="2" align="center"><a href="https://intel.github.io/neural-compressor/latest/docs/source/api-doc/apis.html">APIs</a></td>
      <td colspan="1" align="center"><a href="./docs/source/3x/llm_recipes.md">LLMs Recipes</a></td>
      <td colspan="1" align="center"><a href="./examples/3.x_api/README.md">Examples</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="8">PyTorch Extension APIs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="2" align="center"><a href="./docs/source/3x/PyTorch.md">Overview</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_DynamicQuant.md">Dynamic Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_StaticQuant.md">Static Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_SmoothQuant.md">Smooth Quantization</a></td>
    </tr>
    <tr>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_WeightOnlyQuant.md">Weight-Only Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_FP8Quant.md">FP8 Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_MXQuant.md">MX Quantization</a></td>
        <td colspan="2" align="center"><a href="./docs/source/3x/PT_MixedPrecision.md">Mixed Precision</a></td>
    </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Tensorflow Extension APIs</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="3" align="center"><a href="./docs/source/3x/TensorFlow.md">Overview</a></td>
          <td colspan="3" align="center"><a href="./docs/source/3x/TF_Quant.md">Static Quantization</a></td>
          <td colspan="2" align="center"><a href="./docs/source/3x/TF_SQ.md">Smooth Quantization</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Transformers-like APIs</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/3x/transformers_like_api.md">Overview</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">Other Modules</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="4" align="center"><a href="./docs/source/3x/autotune.md">Auto Tune</a></td>
          <td colspan="4" align="center"><a href="./docs/source/3x/benchmark.md">Benchmark</a></td>
      </tr>
  </tbody>
</table>

> **Note**:
> From 3.0 release, we recommend to use 3.X API. Compression techniques during training such as QAT, Pruning, Distillation only available in [2.X API](https://github.com/intel/neural-compressor/blob/master/docs/source/2x_user_guide.md) currently.

## Selected Publications/Events

* EMNLP'2024: [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) (Sep 2024)
* Blog on Medium: [Quantization on Intel Gaudi Series AI Accelerators](https://medium.com/intel-analytics-software/intel-neural-compressor-v3-0-a-quantization-tool-across-intel-hardware-9856adee6f11) (Aug 2024)
* Blog by Intel: [Neural Compressor: Boosting AI Model Efficiency](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Neural-Compressor-Boosting-AI-Model-Efficiency/post/1604740) (June 2024)
* Blog by Intel: [Optimization of Intel AI Solutions for Alibaba Cloud’s Qwen2 Large Language Models](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-ai-solutions-accelerate-alibaba-qwen2-llms.html) (June 2024)
* Blog by Intel: [Accelerate Meta* Llama 3 with Intel AI Solutions](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-meta-llama3-with-intel-ai-solutions.html) (Apr 2024)
* EMNLP'2023 (Under Review): [TEQ: Trainable Equivalent Transformation for Quantization of LLMs](https://openreview.net/forum?id=iaI8xEINAf&referrer=%5BAuthor%20Console%5D) (Sep 2023)
* arXiv: [Efficient Post-training Quantization with FP8 Formats](https://arxiv.org/abs/2309.14592) (Sep 2023)
* arXiv: [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) (Sep 2023)

> **Note**:
> View [Full Publication List](https://github.com/intel/neural-compressor/blob/master/docs/source/publication_list.md).

## Additional Content

* [Release Information](./docs/source/releases_info.md)
* [Contribution Guidelines](./docs/source/CONTRIBUTING.md)
* [Legal Information](./docs/source/legal_information.md)
* [Security Policy](SECURITY.md)

## Communication
- [GitHub Issues](https://github.com/intel/neural-compressor/issues): mainly for bug reports, new feature requests, question asking, etc.
- [Email](mailto:inc.maintainers@intel.com): welcome to raise any interesting research ideas on model compression techniques by email for collaborations.
- [Discord Channel](https://discord.com/invite/Wxk3J3ZJkU): join the discord channel for more flexible technical discussion.
- [WeChat group](/docs/source/imgs/wechat_group.jpg): scan the QA code to join the technical discussion.

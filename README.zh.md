<div align="center">

Intel® Neural Compressor
===========================
<h3> 支持主流深度学习框架（PyTorch 和 JAX）中常用模型压缩技术的开源 Python 库</h3>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/intel/neural-compressor)
[![version](https://img.shields.io/badge/release-3.10-green)](https://github.com/intel/neural-compressor/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/neural-compressor/blob/main/LICENSE)
[![coverage](https://img.shields.io/badge/coverage-85%25-green)](https://github.com/intel/neural-compressor)
[![Downloads](https://static.pepy.tech/personalized-badge/neural-compressor?period=total&units=international_system&left_color=grey&right_color=green&left_text=downloads)](https://pepy.tech/project/neural-compressor)

[架构设计](./docs/source/design.md#architecture)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[工作流](./docs/source/design.md#workflows)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[官方文档](https://intel.github.io/neural-compressor)

---
<div align="left">

Intel® Neural Compressor 旨在提供主流的模型压缩技术，如静态量化（Static Quantization）、动态量化（Dynamic Quantization）、SmoothQuant、仅权重量化（Weight-Only Quantization）、量化感知训练（Quantization-Aware Training）、混合精度（Mixed Precision）等。

* 通过与 [AutoRound](https://github.com/intel/auto-round) 的深度集成，支持大语言模型（LLM）与视觉-语言模型（VLM）（如 LLaMA、Qwen、DeepSeek、Flux、FramePack 等）跨多种量化技术和低精度数据类型的前沿量化。

* 经过广泛严格测试，全面支持各类 Intel 硬件，如 [Intel Gaudi AI 加速器](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html)、[Intel Core Ultra 处理器](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html)、[Intel Xeon 可扩展处理器](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html)、[Intel Xeon CPU Max 系列](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html)、[Intel 数据中心 GPU Flex 系列](https://www.intel.com/content/www/us/en/products/overview.html) 以及 [Intel 数据中心 GPU Max 系列](https://www.intel.com/content/www/us/en/products/overview.html)；同时提供针对 AMD CPU、ARM CPU 以及 NVIDIA GPU 的基础测试支持。

## 最新动态
* [2026/03] 支持针对 [Keras/JAX](./docs/source/JAX.md) 的 FP8 量化（实验性功能）
* [2026/03] 通过 [AutoRound](./docs/source/PT_AutoRound.md) 支持 FP8 KV 缓存/注意力静态量化（实验性功能）
* [2025/12] 实验性支持 [NVFP4 量化](./docs/source/PT_NVFP4Quant.md)
* [2025/10] 实验性支持 [MXFP8 / MXFP4 量化](./docs/source/PT_MXQuant.md)
* [2025/09] 在 Intel Gaudi AI 加速器上支持 FP8 动态量化（涵盖 Linear、FusedMoE）
* [2025/05] 在 Intel Gaudi AI 加速器上实现 DeepSeek V3/R1 模型的 FP8 静态量化
* [2025/03] 在 Intel CPU/GPU 上通过类似 transformers 的 API 支持 VLM 量化

## 安装指南
请根据您的部署环境选择并安装相应的框架依赖。
### 按需安装 PyTorch 后端框架
Intel Neural Compressor 支持配备 CPU、GPU 和 HPU 的 PyTorch 环境。请根据您的硬件环境安装对应的 PyTorch 版本。
* [安装面向 CPU 的 intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)
* [安装面向 Intel GPU 的 intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)
* [使用针对 HPU 预装 torch 的 Docker 镜像](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click)
  **注意**：Intel Neural Compressor 与 Gaudi 软件栈之间存在版本映射关系，请参阅[版本对应表](./docs/source/gaudi_version_map.md)并确保使用相互匹配的组合版本。
* [安装面向其他平台的 torch](https://pytorch.org/get-started/locally)

### 从 PyPI 安装 Neural Compressor
```
# 框架扩展 API + PyTorch 依赖
pip install neural-compressor-pt
# 框架扩展 API + JAX 依赖
pip install neural-compressor-jax
```
**注意**：更多安装方式请参阅[安装指南](./docs/source/installation_guide.md)。如遇问题，请查看我们的[常见问题解答（FAQ）](./docs/source/faq.md)。

## 快速上手
成功安装相关软件包后，即可开始尝试您的第一个量化程序。**以下示例代码演示了 FP8 量化**，该功能受 Intel Gaudi2 AI 加速器支持。
如需在 Intel Gaudi2 上运行体验，推荐使用包含 Gaudi 软件栈的 Docker 镜像。请参考以下脚本进行环境搭建，更多详情可参阅 [Gaudi 指南](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#launch-docker-image-that-was-built)。

以交互式 Shell 启动容器，[更多信息](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#docker-installation)
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest
```

> 注意：自 Habana 软件版本 >= 1.21.0 起，默认设置为 `PT_HPU_LAZY_MODE=0`。然而，多数低精度函数（如 `convert_from_uint4`）并不支持该设置。因此，我们建议设置 `PT_HPU_LAZY_MODE=1` 以保持兼容性。

运行示例代码：
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

# 用户自定义校准数据。以下为模拟校准示例
model(torch.randn(1, 3, 224, 224).to("hpu"))

model = convert(model)

output = model(torch.randn(1, 3, 224, 224).to("hpu")).to("cpu")
print(output.shape)
```
更多内容请查看 [FP8 量化文档](./docs/source/PT_FP8Quant.md)。

**以下示例代码演示了在 Intel Gaudi2 AI 加速器上加载仅权重（Weight-Only）量化的大语言模型**。
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
**注意：** Intel Neural Compressor 在首次加载时会将模型格式从 auto-gptq 转换为 hpu 格式，并将 hpu_model.safetensors 保存到本地缓存目录以供下次加载使用。因此首次加载可能需要耗费一定时间。

## 文档导航

<table class="docutils">
  <thead>
  <tr>
    <th colspan="8">概述 (Overview)</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2" align="center"><a href="./docs/source/design.md#architecture">架构设计</a></td>
      <td colspan="2" align="center"><a href="./docs/source/design.md#workflows">工作流</a></td>
      <td colspan="2" align="center"><a href="https://intel.github.io/neural-compressor/latest/docs/source/api-doc/apis.html">API 列表</a></td>
      <td colspan="2" align="center"><a href="./examples/README.md">示例代码</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="8">PyTorch 扩展 API</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="8" align="center"><a href="./docs/source/PyTorch.md">概述</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><a href="./docs/source/PT_DynamicQuant.md">动态量化</a></td>
        <td colspan="3" align="center"><a href="./docs/source/PT_StaticQuant.md">静态量化</a></td>
        <td colspan="2" align="center"><a href="./docs/source/PT_SmoothQuant.md">平滑量化 (SmoothQuant)</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><a href="./docs/source/PT_WeightOnlyQuant.md">仅权重量化</a></td>
        <td colspan="3" align="center"><a href="./docs/source/PT_FP8Quant.md">FP8 量化</a></td>
        <td colspan="2" align="center"><a href="./docs/source/PT_MixedPrecision.md">混合精度</a></td>
    </tr>
    <tr>
        <td colspan="4" align="center"><a href="./docs/source/PT_MXQuant.md">MX 量化</a></td>
        <td colspan="4" align="center"><a href="./docs/source/PT_NVFP4Quant.md">NVFP4 量化</a></td>
    </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">类 Transformers API</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/transformers_like_api.md">概述</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">JAX 扩展 API</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/JAX.md">概述</a></td>
      </tr>
  </tbody>
  <thead>
      <tr>
        <th colspan="8">其他模块</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/autotune.md">自动调优 (Auto Tune)</a></td>
      </tr>
  </tbody>
</table>

## 精选论文发表与行业动态

* arXiv: [Faster Inference of LLMs using FP8 on the Intel Gaudi](https://arxiv.org/abs/2503.09975) (Mar 2025)
* PyTorch landscape: [PyTorch general optimizations](https://landscape.pytorch.org/) (Mar 2025)
* Blog on SqueezeBits: [[Intel Gaudi] #4. FP8 Quantization](https://blog.squeezebits.com/intel-gaudi-4-fp8-quantization--40269) (Jan 2025)

> **说明**：
> 查看[完整论文发表列表](https://github.com/intel/neural-compressor/blob/main/docs/source/publication_list.md)。

## 更多资源

* [贡献指南](./docs/source/CONTRIBUTING.md)
* [法律信息](./docs/source/legal_information.md)
* [安全策略](SECURITY.md)

## 交流与反馈
- [GitHub Issues](https://github.com/intel/neural-compressor/issues)：主要用于缺陷反馈、新功能需求提议与技术提问等。
- [电子邮件](mailto:inc.maintainers@intel.com)：欢迎通过邮件提出关于模型压缩技术的各类有趣研究设想以开展深度合作。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月31日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！

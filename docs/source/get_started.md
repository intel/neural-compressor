# Getting Started

1. [Quick Samples](#quick-samples)

2. [Feature Matrix](#feature-matrix)

## Quick Samples

```shell
# Install Intel Neural Compressor
pip install neural-compressor-pt
```
```python
from transformers import AutoModelForCausalLM
from neural_compressor.torch.quantization import RTNConfig, prepare, convert

user_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
quant_config = RTNConfig()
prepared_model = prepare(model=user_model, quant_config=quant_config)
quantized_model = convert(model=prepared_model)
```

## Feature Matrix
Intel Neural Compressor extends PyTorch, TensorFlow and JAX's APIs to support compression techniques.
The below table provides a quick overview of the APIs available in Intel Neural Compressor 3.X.
The project mainly focuses on quantization-related features, especially for algorithms that benefit LLM accuracy and inference.
It also provides some common modules across different frameworks. For example, Auto-tune support accuracy driven quantization and mixed precision, benchmark aimed to measure the multiple instances performance of the quantized model.

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
      <td colspan="2" align="center"><a href="./examples/README.md">Examples</a></td>
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
        <th colspan="8">JAX Extension APIs</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td colspan="8" align="center"><a href="./docs/source/JAX.md">Overview</a></td>
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
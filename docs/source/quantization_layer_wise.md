Layer Wise Quantization (LWQ)
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)

## Introduction

Large language models (LLMs) have shown exceptional performance across various tasks, meanwhile, the substantial parameter size poses significant challenges for deployment. Layer-wise quantization(LWQ) can greatly reduce the memory footprint of LLMs, usually 80-90% reduction, which means that users can quantize LLMs even on single node using GPU or CPU. We can quantize the model under memory-constrained devices, therefore making the huge-sized LLM quantization possible.

<img src="./imgs/lwq_ort.png" width=900 height=400>

*Figure 1: The process of layer-wise quantization for ONNX model. The graph of LLM is split into several parts, and each subgraph is quantized in turn.*

## Supported Framework Model Matrix


<table class="tg">
<thead>
  <tr>
    <th colspan="2" style="text-align:center;vertical-align:middle">Types/Framework</th>
    <th style="text-align:center;vertical-align:middle">ONNX Runtime</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align:center;vertical-align:middle" colspan="2">W8A8 Post Training Static Quantization</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="3">Weight-only Quantization</td>
    <td style="text-align:center;vertical-align:middle">RTN</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td></td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle">AWQ</td>
    <td style="text-align:center;vertical-align:middle">&#10005;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle">GPTQ</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
  </tr>
</tbody>
</table>

## Examples

```python
from neural_compressor_ort.quantization import matmul_4bits_quantizer
algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(layer_wise_quant=True)
quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
    model,
    algo_config=algo_config,
)
quant.process()
qmodel = quant.model
```

Layer Wise Quantization (LWQ)
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)

## Introduction

Large language models (LLMs) have shown exceptional performance across various tasks, meanwhile, the substantial parameter size poses significant challenges for deployment. Layer-wise quantization(LWQ) can greatly reduce the memory footprint of LLMs, usually 80-90% reduction, which means that users can quantize LLMs even on single node using GPU or CPU. We can quantize the model under memory-constrained devices, therefore making the huge-sized LLM quantization possible.

<img src="./imgs/lwq.png" width=780 height=429>

*Figure 1: The process of layer-wise quantization for PyTorch model. The color grey means empty parameters and the color blue represents parameters need to be quantized. Every rectangle inside model represents one layer.*

<img src="./imgs/lwq_ort.png" width=900 height=400>

*Figure 2: The process of layer-wise quantization for ONNX model. The graph of LLM is split into several parts, and each subgraph is quantized in turn.*

## Supported Framework Model Matrix


<table class="tg">
<thead>
  <tr>
    <th colspan="2" style="text-align:center;vertical-align:middle">Types/Framework</th>
    <th style="text-align:center;vertical-align:middle">PyTorch</th>
    <th style="text-align:center;vertical-align:middle">ONNX Runtime</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align:center;vertical-align:middle" colspan="2">W8A8 Post Training Static Quantization</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle" rowspan="4">Weight-only Quantization</td>
    <td style="text-align:center;vertical-align:middle">RTN</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
    <td style="text-align:center;vertical-align:middle" rowspan="4">&#10005;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle">AWQ</td>
    <td style="text-align:center;vertical-align:middle">&#10005;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle">GPTQ</td>
    <td style="text-align:center;vertical-align:middle">&#10004;</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle">TEQ</td>
    <td style="text-align:center;vertical-align:middle">&#10005;</td>
  </tr>
</tbody>
</table>

## Examples

#### PyTorch framework example

```python
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor.torch_utils.layer_wise_quant import load_empty_model

fp32_model = load_empty_model(model_name_or_path, torchscript=True)
conf = PostTrainingQuantConfig(
    approach="weight_only",
    recipes={
        "layer_wise_quant": True,
        "rtn_args": {"enable_full_range": True},
    },
)

q_model = quantization.fit(
    fp32_model,
    conf,
    calib_dataloader=eval_dataloader,
    eval_func=lambda x: 0.1,
)
ouput_dir = "./saved_model"
q_model.save(ouput_dir)
q_model = load(ouput_dir, fp32_model, weight_only=True, layer_wise=True)
```

#### ONNX Runtime framework example

```python
from neural_compressor import quantization, PostTrainingQuantConfig

conf = PostTrainingQuantConfig(recipes={"layer_wise_quant": True})
q_model = quantization.fit(fp32_model_path, conf, calib_dataloader=dataloader)
q_model.save(int8_model_path)
```

Refer to [ONNX Runtime llama-2 LWQ example](../../examples/onnxrt/nlp/huggingface_model/text_generation/llama/quantization/weight_only)

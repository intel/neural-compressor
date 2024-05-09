Microscaling Quantization
===============

1. [Introduction](#introduction)
2. [Supported Framework Model Matrix](#supported-framework-model-matrix)
3. [Get Started with Microscaling Quantization API](#get-start-with-microscaling-quantization-api)
4. [Examples](#examples)
5. [Reference](#reference)

## Introduction

Numerous breakthroughs have emerged across various fields, such as text analysis, language translation and chatbot technologies, fueled by the development of large language models (LLMs). Nevertheless, their increasing power comes with the challenge of explosive growth in parameters, posing obstacles for practical use. To balance memory limits and accuracy preservation for AI models, the Microscaling (MX) specification was promoted from the well-known Microsoft Floating Point (MSFP) data type [1, 2]:

<a target="_blank" href="./imgs/mx_format.png" text-align:center>
    <center> 
        <img src="./imgs/mx_format.png" alt="Definition of MX data type (source [2])" height=200> 
    </center>
</a>

At an equivalent accuracy level, the MX data type demonstrates the ability to occupy a smaller area and incur lower energy costs for multiply-accumulate compared to other conventional data types on the same silicon [1].

Neural Compressor seamlessly applies the MX data type to post-training quantization, offering meticulously crafted recipes to empower users to quantize LLMs without sacrificing accuracy. The workflow is shown as below.

<a target="_blank" href="./imgs/mx_workflow.png" text-align:center>
    <center> 
        <img src="./imgs/mx_workflow.png" alt="Workflow of MX Quant (source [3])" height=200> 
    </center>
</a>

The memory and computational limits of LLMs are more severe than other general neural networks, so our exploration focuses on LLMs first. The following table shows the basic MX quantization recipes in Neural Compressor and enumerates distinctions among various data types. The MX data type replaces general float scale with powers of two to be more hardware-friendly. It adapts a granularity falling between per-channel and per-tensor to balance accuracy and memory consumption.

|            | MX Format |  INT8  |  FP8  |
|------------|--------------|------------|------------|
|  Scale  |   $2^{exp}$   |  $\frac{MAX}{amax}$  |  $\frac{MAX}{amax}$  |
|  Zero point  |   0 (None)   | $2^{bits - 1}$ or $-min * scale$ |   0 (None)   |
|  Granularity  |  per-block (default blocksize is 32)   |  per-channel or per-tensor  | per-tensor  |

The exponent (exp) is equal to torch.floor(torch.log2(amax)), MAX is the representation range of the data type, amax is the max absolute value of per-block tensor, and rmin is the minimum value of the per-block tensor.


## Supported Framework Model Matrix

|   PyTorch  | ONNX Runtime | TensorFlow |
|------------|--------------|------------|
|  &#10004;  |   &#10005;   |  &#10005;  |


## Get Started with Microscaling Quantization API

To get a model quantized with Microscaling Data Types, users can use the Microscaling Quantization API as follows.

```python
from neural_compressor.torch.quantization import MXQuantConfig, quantize
quant_config = MXQuantConfig(w_dtype=args.w_dtype, act_dtype=args.act_dtype, weight_only=args.woq)
user_model = quantize(model=user_model, quant_config=quant_config)
```
  
## Examples

- PyTorch [huggingface models](/examples/3.x_api/pytorch/nlp/huggingface_models/language-modeling/quantization/mx)


## Reference

[^1]: Darvish Rouhani, Bita, et al. "Pushing the limits of narrow precision inferencing at cloud scale with microsoft floating point." Advances in neural information processing systems 33 (2020): 10271-10281 

[^2]: OCP Microscaling Formats (MX) Specification

[^3]: Rouhani, Bita Darvish, et al. "Microscaling Data Formats for Deep Learning." arXiv preprint arXiv:2310.10537 (2023). 
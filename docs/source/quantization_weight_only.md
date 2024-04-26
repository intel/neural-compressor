Weight Only Quantization (WOQ)
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)

4. [WOQ Algorithms Tuning](#woq-algorithms-tuning)


## Introduction

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy.  Compared to normal quantization like W8A8,  weight only quantization is probably a better trade-off to balance the performance and the accuracy, since we will see below that the bottleneck of deploying LLMs is the memory bandwidth and normally weight only quantization could lead to better accuracy.

Model inference: Roughly speaking , two key steps are required to get the model's result. The first one is moving the model from the memory to the cache piece by piece, in which, memory bandwidth $B$ and parameter count $P$ are the key factors, theoretically the time cost is  $P*4 /B$. The second one is  computation, in which, the device's computation capacity  $C$  measured in FLOPS and the forward FLOPs $F$ play the key roles, theoretically the cost is $F/C$.

Text generation:  The most famous application of LLMs is text generation, which predicts the next token/word  based on the inputs/context. To generate a sequence of texts, we need to predict them one by one. In this scenario,  $F\approx P$  if some operations like bmm are ignored and past key values have been saved. However, the  $C/B$ of the modern device could be to **100X,** that makes the memory bandwidth as the bottleneck in this scenario.

Besides, as mentioned in many papers[1][2], activation quantization is the main reason to cause the accuracy drop. So for text generation task,  weight only quantization is a preferred option in most cases.

Theoretically, round-to-nearest (RTN) is the most straightforward way to quantize weight using scale maps. However, when the number of bits is small (e.g. 3), the MSE loss is larger than expected. A group size is introduced to reduce elements using the same scale to improve accuracy.

There are many excellent works for weight only quantization to improve its accuracy performance, such as AWQ[3], GPTQ[4]. Neural compressor integrates these popular algorithms in time to help customers leverage them and deploy them to their own tasks.


## Supported Framework Model Matrix

| Algorithms/Framework |   ONNX Runtime  |
|--------------|----------|
|       RTN      |  &#10004;  |
|       AWQ      |  &#10004;  |
|      GPTQ      | &#10004; |

> **RTN:** A quantification method that we can think of very intuitively. It does not require additional datasets and is a very fast quantization method. Generally speaking, RTN will convert the weight into a uniformly distributed integer data type, but some algorithms, such as Qlora, propose a non-uniform NF4 data type and prove its theoretical optimality.

> **GPTQ:** A new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly efficient[4]. The weights of each column are updated based on the fixed-scale pseudo-quantization error and the inverse of the Hessian matrix calculated from the activations. The updated columns sharing the same scale may generate a new max/min value, so the scale needs to be saved for restoration.

> **AWQ:** Proved that protecting only 1% of salient weights can greatly reduce quantization error. the salient weight channels are selected by observing the distribution of activation and weight per channel. The salient weights are also quantized after multiplying a big scale factor before quantization for preserving.

## Examples
### **Quantization Capability**

| Config | Capability |
|---|---|
| weight_dtype | ['int'] |
| weight_bits | [1, ..., 8] |
| weight_group_size | [-1, 1, ..., $C_{in}$] |
| weight_sym | ['asym', 'sym'] |
| algorithm | ['RTN', 'AWQ', 'GPTQ'] |

Notes:
*weight_group_size = -1* refers to **per output channel quantization**. Taking a MatMul operator (input channel = $C_{in}$, output channel = $C_{out}$) for instance, when *weight_group_size = -1*, quantization will calculate total $C_{out}$ quantization parameters. Otherwise, when *weight_group_size = gs* quantization parameters are calculate with every $gs$ elements along with the input channel, leading to total $C_{out} \times (C_{in} / gs)$ quantization parameters.


**RTN arguments**

|  rtn_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  accuracy_level |      0     | Support 0 (unset), 1(fp32 compute type of jblas kernel), 2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel), 4 (int8 compute type of jblas kernel) |
|  ratios |      {}     | Percentile of clip   |
|  providers |      ["CPUExecutionProvider"]     | Execution providers to use   |
|  layer_wise_quant    |      False    | Whether to quantize model layer by layer to save memory footprint. |
|  quant_last_matmul |      True     |  Whether to quantize the last matmul of the model   |


**AWQ arguments**

|  awq_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  accuracy_level |      0     | Support 0 (unset), 1(fp32 compute type of jblas kernel), 2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel), 4 (int8 compute type of jblas kernel) |
|  enable_auto_scale |      True     | Whether to search for best scales based on activation distribution   |
|  enable_mse_search |      True     | Whether to search for the best clip range from range [0.91, 1.0, 0.01] |
|  providers |      ["CPUExecutionProvider"]     | Execution providers to use   |
|  quant_last_matmul |      True     |  Whether to quantize the last matmul of the model   |

**GPTQ arguments**

|  gptq_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  accuracy_level |      0     | Support 0 (unset), 1(fp32 compute type of jblas kernel), 2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel), 4 (int8 compute type of jblas kernel) |
|  percdamp | 0.01 | Percentage of Hessian's diagonal values' average, which will be added to Hessian's diagonal to increase numerical stability|
|  block_size  | 128 | Execute GPTQ quantization per block, block shape = [$C_{out}$, block_size] |
|  actorder | False |   Whether to sort Hessian's diagonal values to rearrange channel-wise quantization order|
|  mse | False |   Whether get scale and zero point with mse error |
|  perchannel | True |   Whether quantize weight per-channel |
|  providers |      ["CPUExecutionProvider"]     | Execution providers to use   |
|  layer_wise_quant    |      False    | Whether to quantize model layer by layer to save memory footprint. |
|  quant_last_matmul |      True     |  Whether to quantize the last matmul of the model   |

**Note:** Neural compressor provides `Unsigned integer for asymmetric quantization` and `Signed integer for symmetric quantization`. Please follow the below section to compress the low bit data type for saving.


### **User Code Example**
```python
from neural_compressor_ort.quantization import matmul_4bits_quantizer

algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(calibration_data_reader=calibration_data_reader)
quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
    model,
    block_size=32,
    is_symmetric=False,
    algo_config=algo_config,
)
quant.process()
q_model = quant.model
```

## WOQ Algorithms Tuning

To find the best algorithm, users can leverage the `autotune` feature to explore a set of configurations. It automatically searches for the optimal one with the best result. Users have the option to specify their own list of potential configurations or utilize the pre-defined configuration set.

**Pre-defined configurations**

| WOQ configurations | Comments |
|------------------|-------|
|RTN_G32ASYM| {"algorithm": "RTN", "group_size": 32, "scheme": "asym"}|
|GPTQ_G32ASYM| {"algorithm": "GPTQ", "group_size": 32, "scheme": "asym"}|
|GPTQ_G32ASYM_DISABLE_LAST_MATMUL| {"algorithm": "GPTQ", "group_size": 32, "scheme": "asym"} <br> & disable last MatMul|
|GPTQ_G128ASYM| {"algorithm": "GPTQ", "group_size": 128, "scheme": "asym"}|
|AWQ_G32ASYM| {"algorithm": "AWQ", "group_size": 32, "scheme": "asym"}|

### **User code example**

```python
from neural_compressor_ort.quantization import get_woq_tuning_config, autotune
from neural_compressor_ort.common.base_tuning import TuningConfig

tune_config = TuningConfig(config_set=get_woq_tuning_config())
best_model = autotune(
    model_input=model,
    tune_config=tune_config,
    eval_fn=eval_fn,
    calibration_data_reader=data_reader,
)
```

Refer to this [link](../../examples/onnxrt/nlp/huggingface_model/text_generation/llama/quantization/weight_only) for an example of WOQ algorithms tuning on ONNX Llama models.



## Reference

[1]. Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022).

[2]. Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).

[3]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).

[4]. Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

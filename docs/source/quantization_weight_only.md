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

| Algorithms/Framework |   PyTorch  |    ONNX Runtime    |
|--------------|----------|----------|
|       RTN      |  &#10004;  |  &#10004;  |
|       AWQ      |  &#10004;  | &#10004; |
|      GPTQ      | &#10004; | &#10004; |
|      TEQ      | &#10004; | stay tuned |

**Note:** To get the validated accuracy results on popular models, please refer to [PyTorch Models with Torch 2.0.1+cpu in WOQ Mode](./validated_model_list.md/#pytorch-models-with-torch-201cpu-in-woq-mode)

> **RTN:** A quantification method that we can think of very intuitively. It does not require additional datasets and is a very fast quantization method. Generally speaking, RTN will convert the weight into a uniformly distributed integer data type, but some algorithms, such as Qlora, propose a non-uniform NF4 data type and prove its theoretical optimality.

> **GPTQ:** A new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly efficient[4]. The weights of each column are updated based on the fixed-scale pseudo-quantization error and the inverse of the Hessian matrix calculated from the activations. The updated columns sharing the same scale may generate a new max/min value, so the scale needs to be saved for restoration.

> **AWQ:** Proved that protecting only 1% of salient weights can greatly reduce quantization error. the salient weight channels are selected by observing the distribution of activation and weight per channel. The salient weights are also quantized after multiplying a big scale factor before quantization for preserving.

> **TEQ:** A trainable equivalent transformation that preserves the FP32 precision in weight-only quantization. It is inspired by AWQ while providing a new solution to search for the optimal per-channel scaling factor between activations and weights.

## Examples
### **Quantization Capability**

| Config | Capability |
|---|---|
| dtype | ['int', 'nf4', 'fp4'] |
| bits | [1, ..., 8] |
| group_size | [-1, 1, ..., $C_{in}$] |
| scheme | ['asym', 'sym'] |
| algorithm | ['RTN', 'AWQ', 'GPTQ'] |

Notes:
- *group_size = -1* refers to **per output channel quantization**. Taking a linear layer (input channel = $C_{in}$, output channel = $C_{out}$) for instance, when *group size = -1*, quantization will calculate total $C_{out}$ quantization parameters. Otherwise, when *group_size = gs* quantization parameters are calculate with every $gs$ elements along with the input channel, leading to total $C_{out} \times (C_{in} / gs)$ quantization parameters.
- 4-bit NormalFloat(NF4) is proposed in QLoRA[5]. 'fp4' includes [fp4_e2m1](../../neural_compressor/adaptor/torch_utils/weight_only.py#L37) and [fp4_e2m1_bnb](https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/functional.py#L735). By default, fp4 refers to fp4_e2m1_bnb.

**RTN arguments**

|  rtn_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  enable_full_range |      False     |   Whether to use -2**(bits-1) in sym scheme  |
|  enable_mse_search |      False     | Whether to search for the best clip range from range [0.805, 1.0, 0.005] |
|  return_int |      False     | Whether to return compressed model with torch.int32 data type |
|  group_dim  |       1       |   0 means splitting output channel, 1 means splitting input channel   |

**AWQ arguments**

|  awq_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  enable_auto_scale |      True     | Whether to search for best scales based on activation distribution   |
|  enable_mse_search |      True     | Whether to search for the best clip range from range [0.91, 1.0, 0.01] |
|  folding   |      False    | False will allow insert mul before linear when the scale cannot be absorbed by last layer, else won't |

**GPTQ arguments**

|  gptq_args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  actorder | False |   Whether to sort Hessian's diagonal values to rearrange channel-wise quantization order|
|  percdamp | 0.01 | Percentage of Hessian's diagonal values' average, which will be added to Hessian's diagonal to increase numerical stability|
|  nsamples  | 128 |  Calibration samples' size |
|  pad_max_length  | 2048 | Whether to align calibration data to a fixed length. This value should not exceed model's acceptable sequence length. Please refer to  model's config json to find out this value.|
|  use_max_length  | False | Whether to align all calibration data to fixed length, which equals to pad_max_length. |
|  block_size  | 128 | Execute GPTQ quantization per block, block shape = [$C_{out}$, block_size] |
|  static_groups  | False | Whether to calculate group wise quantization parameters in advance. This option mitigate actorder's extra computational requirements |
|  true_sequential  | False | Whether to quantize layers within a transformer block in their original order. This can lead to higher accuracy but slower overall quantization process. |
|  lm_head  | False | Whether to quantize the lm_head (linear layer related to prediction in the end of the language models). |

**Note:** Neural compressor provides `Unsigned integer for asymmetric quantization` and `Signed integer for symmetric quantization`. Please follow the below section to compress the low bit data type for saving.

### **Export Compressed Model**
To support low memory inference, Neural Compressor implemented WeightOnlyLinear, a torch.nn.Module, to compress the fake quantized fp32 model. Since torch does not provide flexible data type storage, WeightOnlyLinear combines low bits data into a long date type, such as torch.int8 and torch.int32. Low bits data includes weights and zero points. When using WeightOnlyLinear for inference, it will restore the compressed data to float32 and run torch linear function.

**Export arguments**

| export args  | default value |                               comments                              |
|----------|-------------|-------------------------------------------------------------------|
|  use_optimum_format  |     True       |  Whether to use the popular format used in [Optimum](https://github.com/huggingface/optimum/blob/e0927976d06d163ed09fe5bd80d013e1cfa0c463/docs/source/llm_quantization/usage_guides/quantization.mdx#L5)  |
|  sym_full_range |      False     | Whether to leverage the full compression range under symmetric quantization |
|  compression_dtype  |       torch.int32       |  Data type for compressed dtype, select from [torch.int8\|16\|32\|64]. It's torch.int32 when use_optimum_format=True |
|  compression_dim  |       1       |   0 means output channel while 1 means input channel. It's 1 for weight and 0 for zero-point when use_optimum_format=True   |
|  scale_dtype  |       torch.float32       |  Data type for scale and bias. It's torch.float16 when use_optimum_format=True   |
| qweight_config_path |      None     |  set the path of qconfig.json if you want to export model with json file |
| gptq_config_path |      None     |  If need to export model with fp32_model and json file, set the path of gptq_config.json for GPTQ quantized model|

**Note:** The format used in Optimum is acceptable for transformers, which makes it easy to use. However, this format is rather special, the main differences are as follows:

> 1: Compression Dimension: weight = 1, zero = 0 and both are transposed.
> 2: Zero Point: zero_point-= 1 before compression. zero_point is always required even for sym.
> 3: Group Index: Use the same number for a group instead of recording channel order.


### **User Code Example**
```python
conf = PostTrainingQuantConfig(
    approach="weight_only",
    op_type_dict={
        ".*": {  # re.match
            "weight": {
                "bits": 8,  # 1-8 bit
                "group_size": -1,  # -1 (per-channel)
                "scheme": "sym",
                "algorithm": "RTN",
            },
        },
    },
    recipes={
        # 'rtn_args':{'enable_full_range': True, 'enable_mse_search': True},
        # 'gptq_args':{'percdamp': 0.01, 'actorder':True, 'block_size': 128, 'nsamples': 128, 'use_full_length': False},
        # 'awq_args':{'enable_auto_scale': True, 'enable_mse_search': True, 'n_blocks': 5},
    },
)
q_model = quantization.fit(model, conf, eval_func=eval_func)
q_model.save("saved_results")
compressed_model = q_model.export_compressed_model()
torch.save(compressed_model.state_dict(), "compressed_model.pt")
# or
model = Model()
compressed_model = export_compressed_model(
    model,
    saved_dir="saved_results",
)
```

The saved_results folder contains two files: `best_model.pt` and `qconfig.json`, and the generated q_model is a fake quantized model.

To seek the performance of weight-only quantized models, Please go to [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/text-generation/quantization#1-performance) to quantize and deploy the model.

## WOQ Algorithms Tuning

To find the best algorithm, users can omit specifying a particular algorithm. In comparison to setting a specific algorithm, this tuning process will traverse through a set of pre-defined WOQ configurations and identify the optimal one with the best result. For details usage, please refer to the [tuning strategy](./tuning_strategies.md#Basic).

> **Note:** Currently, this behavior is specific to the `ONNX Runtime` backend.

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
conf = PostTrainingQuantConfig(
    approach="weight_only",
    quant_level="auto",  # quant_level supports "auto" or 1 for woq config tuning
)
q_model = quantization.fit(model, conf, eval_func=eval_func, calib_dataloader=dataloader)
q_model.save("saved_results")
```

Refer to this [link](../../examples/onnxrt/nlp/huggingface_model/text_generation/llama/quantization/weight_only) for an example of WOQ algorithms tuning on ONNX Llama models.



## Reference

[1]. Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022).

[2]. Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).

[3]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).

[4]. Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

[5]. Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." arXiv preprint arXiv:2305.14314 (2023).

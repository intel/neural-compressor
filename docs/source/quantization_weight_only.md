Weight Only Quantization
=====

1. [Introduction](#introduction)

2. [Supported Framework Model Matrix](#supported-framework-model-matrix)

3. [Examples](#examples)


## Introduction

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computational demands of these modern architectures while maintaining the accuracy.  Compared to normal quantization like W8A8,  weight only quantization is probably a better trade-off to balance the performance and the accuracy, since we will see below that the bottleneck of deploying LLMs is the memory bandwidth and normally weight only quantization could lead to better accuracy.

Model inference: Roughly speaking , two key steps are required to get the model's result. The first one is moving the model from the memory to the cache piece by piece, in which, memory bandwidth $B$ and parameter count $P$ are the key factors, theoretically the time cost is  $P*4 /B$. The second one is  computation, in which, the device's computation capacity  $C$  measured in FLOPS and the forward FLOPs $F$ play the key roles, theoretically the cost is $F/C$.

Text generation:  The most famous application of LLMs is text generation, which predicts the next token/word  based on the inputs/context. To generate a sequence of texts, we need to predict them one by one. In this scenario,  $F\approx P$  if some operations like bmm are ignored and past key values have been saved. However, the  $C/B$ of the modern device could be to **100X,** that makes the memory bandwidth as the bottleneck in this scenario.

Besides, as mentioned in many papers[1][2], activation quantization is the main reason to cause the accuracy drop. So for text generation task,  weight only quantization is a preferred option in most cases.

Theoretically, round-to-nearest (RTN) is the mose straightforward way to quantize weight using scale maps. However, when the number of bits is small (e.g. 3), the MSE loss is larger than expected. A group size is introduced to reduce elements using the same scale to improve accuracy.

There are many excellent works for weight only quantization to improve its accuracy performance, such as AWQ[3], GPTQ[4]. Neural compressor integrates these popular algorithms in time to help customers leverage them and deploy them to their own tasks.

## Supported Framework Model Matrix

| Algorithms/Framework |   PyTorch  |    ONNX Runtime    |
|:--------------:|:----------:|:----------:|
|       RTN      |  &#10004;  |  &#10004;  |
|       AWQ      |  &#10004;  | &#10004; |
|      GPTQ      | &#10004; | &#10004; |
|      TEQ      | &#10004; | stay tuned |

## Examples
### **Quantization Capability**:
| Config | Capability |
| :---: | :---:|
| bits | [1-8] |
| group_size | [-1, 1-N] | 
| scheme | ['asym', 'sym'] |
| algorithm | ['RTN', 'AWQ'] |

**RTN arguments**:
|  rtn_args  | default value |                               comments                              |
|:----------:|:-------------:|:-------------------------------------------------------------------:|
| sym_full_range |      False     |   Whether use -2**(bits-1) in sym scheme, for example,    |
|  mse_range |      False     | Whether search for the best clip range from range [0.805, 1.0, 0.005] |
|  return_int |      False     | Whether return compressed model with int data type |

**AWQ arguments**:
|  awq_args  | default value |                               comments                              |
|:----------:|:-------------:|:-------------------------------------------------------------------:|
| auto_scale |      True     | Whether search for best scales based on activation distribution   |
|  mse_range |      True     | Whether search for the best clip range from range [0.91, 1.0, 0.01] |
|  folding   |      False    | False will allow insert mul before linear when the scale cannot be absorbed by last layer, else won't |


**Note**: `group_size=-1` indicates the per-channel quantization per output channel. `group_size=[1-N]` indicates splitting the input channel elements per group_size.

### **Export Compressed Model**
To support low memory inference, Neural Compressor implemented WeightOnlyLinear, a torch.nn.Module, to compress the fake quantized fp32 model. Since torch does not provide flexible data type storage, WeightOnlyLinear combines low bits data into a long date type, such as torch.int8 and torch.int32. Low bits data includes weights and zero points. When using WeightOnlyLinear for inference, it will restore the compressed data to float32 and run torch linear function.

**Export arguments**:
| export args  | default value |                               comments                              |
|:----------:|:-------------:|:-------------------------------------------------------------------:|
| qweight_config_path |      None     |  If need to export model with fp32_model and json file, set the path of qconfig.json |
|  sym_full_range |      False     | Whether to leverage the full compression range under symmetric quantization |
|  compression_dtype  |       torch.int32       |  Data type for compressed dtype, select from [torch.int8|16|32|64]   |
|  compression_dim  |       1       |   0 means output channel while 1 means input channel   |
|  scale_dtype  |       torch.float32       |  Data type for scale and bias   |

### **User code**:
```python
conf = PostTrainingQuantConfig(
    approach='weight_only',
    op_type_dict={
        '.*':{ 	# re.match
            "weight": {
                'bits': 8, # 1-8 bit 
                'group_size': -1,  # -1 (per-channel)
                'scheme': 'sym', 
                'algorithm': 'RTN', 
            },
        },
    },
    ### GPTQ is WIP
    recipes={
        # 'gptq_args':{'percdamp': 0.01},
        'awq_args':{'auto_scale': True, 'mse_range': True, 'n_blocks': 5},
    },
)
q_model = quantization.fit(model, conf, eval_func=eval_func)
q_model.save('saved_results')
compressed_model = q_model.export_compressed_model(
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float16,
)
torch.save(compressed_model.state_dict(), 'compressed_model.pt')
```

The saved_results folder contains two files: `best_model.pt` and `qconfig.json`, and the generated q_model is a fake quantized model.

## Reference

[1]. Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022).

[2]. Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).

[3]. Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv preprint arXiv:2306.00978 (2023).

[4]. Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

Quantization on Client
==========================================

1. [Introduction](#introduction)
2. [Get Started](#get-started) \
   2.1 [Get Default Algorithm Configuration](#get-default-algorithm-configuration)\
   2.2 [Optimal Performance and Peak Memory Usage](#optimal-performance-and-peak-memory-usage)


## Introduction

For `RTN`, `GPTQ`, and `Auto-Round`, we offer default algorithm configurations tailored for different processor types (`client` and `sever`). These configurations are specifically designed for both client and server machines, with a lightweight setup optimized for client devices.


## Get Started

### Get Default Algorithm Configuration

We take the `RTN` algorithm as example to demonstrate the usage on a client machine.

```python
from neural_compressor.torch.quantization import get_default_rtn_config, convert, prepare
from neural_compressor.torch.utils import load_empty_model

model_state_dict_path = "/path/to/model/state/dict"
float_model = load_empty_model(model_state_dict_path)
quant_config = get_default_rtn_config()
prepared_model = prepare(float_model, quant_config)
quantized_model = convert(prepared_model)
```

> [!TIP]
> By default, the appropriate configuration is determined based on hardware information, but users can explicitly specify `processor_type` as `client` or `server`.


For Windows machines, simply run the program to automatically utilize all available cores:

```bash
python main.py
```


### Optimal Performance and Peak Memory Usage


- 7B models (e.g., [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)): the quantization process takes about 65 seconds, with a peak memory usage of around 6GB.
- 1.5B models (e.g., [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)),  the quantization process takes about 20 seconds, with a peak memory usage of around 5GB.


> [!TIP]
> For Linux systems, users need to configure the environment variables appropriately to achieve optimal performance. For example, set the `OMP_NUM_THREADS` explicitly, For processors with hybrid architecture (including both P-cores and E-cores), it is recommended to bind tasks to all P-cores using `taskset`.

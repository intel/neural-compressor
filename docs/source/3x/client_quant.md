Quantization on Client
==========================================

1. [Introduction](#introduction)
2. [Get Started](#get-started)

## Introduction

For `RTN`, and `GPTQ` algorithms, we provide default algorithm configurations for different processor types (`client` and `sever`). Generally, lightweight configurations are tailored specifically for client devices to enhance performance and efficiency.


## Get Started

Here, we take the `RTN` algorithm as example to demonstrate the usage on a client machine.

```python
from neural_compressor.torch.quantization import get_default_rtn_config, convert, prepare
from neural_compressor.torch import load_empty_model

model_state_dict_path = "/path/to/model/state/dict"
float_model = load_empty_model(model_state_dict_path)
quant_config = get_default_rtn_config()
prepared_model = prepare(float_model, quant_config)
quantized_model = convert(prepared_model)
```

> [!TIP]
> By default, the appropriate configuration is determined based on hardware information, but users can explicitly specify `processor_type` as either `client` or `server` when calling `get_default_rtn_config`.


For Windows machines, run the following command to utilize all available cores automatically:

```bash
python main.py
```

> [!TIP]
> For Linux systems, users need to configure the environment variables appropriately to achieve optimal performance. For example, set the `OMP_NUM_THREADS` explicitly. For processors with hybrid architecture (including both P-cores and E-cores), it is recommended to bind tasks to all P-cores using `taskset`.

RTN quantization is a quick process, finishing in tens of seconds and using several GB of RAM when working with 7B models, e.g.,[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). However, for the higher accuracy, GPTQ algorithm is recommended, but be prepared for a longer quantization time.

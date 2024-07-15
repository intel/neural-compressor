Quantization on Client
==========================================
1. [Introduction](#introduction)
3. [Get Started](#get-started) \
    2.1 [Get Default Algorithm Configuration](#get-default-algorithm-configuration)\
    2.2 [Optimal Performance](#optimal-performance)


## Introduction

Currently, we support different default algorithm configurations based on the type of processor type for `RTN`, `GPTQ`, and `Auto-Round` on the PyTorch framework. Processors are roughly categorized into client and server types, with a lightweight configuration provided for a machine with client processors.


## Get Started
### Get Default Algorithm Configuration

To obtain the default algorithm configuration, users can either specify the `processor_type` explicitly when calling the configuration API or leave it unspecified. In the latter case, we will automatically determine the appropriate configuration based on hardware information. A machine is identified as a server if it meets one of the following criteria:

- If there is more than one sockets
- If the brand name includes `Xeon`
- If the DRAM size is greater than 32GB

> [!TIP]
> The last criterion may not always be accurate, but models larger than 7B typically require more than 32GB DRAM. We assume that users won't run these models on client machines.

Below is an example to get the default configuration of RTN.

```python
config_by_auto_detect = get_default_rtn_config()
config_for_client = get_default_rtn_config(processor_type="client")
config_for_server = get_default_rtn_config(processor_type="server")
```

### Optimal Performance

#### Windows
On Windows machines, it is recommended to run the application directly. The system will automatically utilize all available cores.

```bash
python ./main.py
```
> [!NOTE]
> - For 7B models, like [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), the quantization process takes about 65 seconds and the peak memory usage is about 6GB.
> - For 1.5B models, like [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct),  the quantization process takes about 20 seconds and the peak memory usage is about 5GB.

### Linux

On Linux machines, users need configure the environment variables appropriately. For example, the 12th Generation and later processors, which is Hybrid Architecture include both P-cores and E-Cores. It is recommended to run the example with all of P-cores to achieve optimal performance.

```bash
# e.g. for Intel® Core™ Ultra 7 Processor 155H, it includes 6 P-cores and 10 E-cores
OMP_NUM_THREADS=12 taskset -c 0-11 python ./main.py
```

> [!NOTE]
> To identify E-cores and P-cores on a Linux system,, please refer [this](https://stackoverflow.com/a/71282744/23445462).



> [!CAUTION]
> Please use `neural_compressor.torch.load_empty_model` to initialize a empty model to reduce the memory usage.

Quantization on Client
==========================================
1. [Introduction](#introduction)
3. [Get Started](#get-started) \
    2.1 [Get Default Algorithm Configuration](#get-default-algorithm-configuration)\
    2.2 [Get Optimal Performance](#get-optimal-performance)


## Introduction

For `RTN`, `GPTQ`, and `Auto-Round` on the PyTorch framework, we offer default algorithm configurations tailored for different processor types. Processors are roughly categorized into client and server types, with a lightweight configuration specifically designed for client machines.


## Get Started
### Get Default Algorithm Configuration

To obtain the default algorithm configuration, users can either specify the `processor_type` explicitly when calling the configuration API or leave it empty. In the latter case, we will automatically determine the appropriate configuration based on hardware information. A machine is identified as a server if it meets one of the following criteria:

- It has more than one socket.
- Its brand name includes `Xeon`.
- Its DRAM size is exceeds 32GB.

> [!TIP]
> The DRAM criterion may not always be accurate. However, models larger than 7B typically require more than 32GB of DRAM, and it is assumed that such models will not be used on client machines.

Here’s an example of how to get the default configuration for `RTN`:

```python
config_by_auto_detect = get_default_rtn_config()
config_for_client = get_default_rtn_config(processor_type="client")
config_for_server = get_default_rtn_config(processor_type="server")
```

### Get Optimal Performance

#### Windows
On Windows machines, simply running the program will allow the system to utilize all available cores automatically.

```bash
python main.py
```
> [!NOTE]
> - For 7B models, such as [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), the quantization process takes about 65 seconds, with a peak memory usage of around 6GB.
> - For 1.5B models, like [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct),  the quantization process takes about 20 seconds, with a peak memory usage of around 5GB.

### Linux

On Linux systems, you need to configure the environment variables appropriately to achieve optimal performance. For instance, with Intel 12th generation and later processors featuring hybrid architecture (including both P-cores and E-cores), it is recommended to bind tasks to all P-cores.

```bash
# e.g. for Intel® Core™ Ultra 7 Processor 155H, it includes 6 P-cores and 10 E-cores
OMP_NUM_THREADS=12 taskset -c 0-11 python main.py
```

> [!NOTE]
> To identify E-cores and P-cores on a Linux system, please refer [this](https://stackoverflow.com/a/71282744/23445462).


> [!CAUTION]
> Please use `neural_compressor.torch.load_empty_model` to initialize a empty model and reduce the memory usage.

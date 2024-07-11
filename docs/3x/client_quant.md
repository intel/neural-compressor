Quantization on Client
==========================================
1. [Introduction](#introduction)
3. [Get Started](#get-started) \
    2.1 [Get Default Algorithm Configuration](#get-default-algorithm-configuration)\
    2.2 [Set Environment Variables for Optimal Performance](#set-environment-variables-for-optimal-performance)


## Introduction

Currently, we support different default algorithm configurations based on the type of processor for `RTN`, `GPTQ`, and `Auto-Round` on the PyTorch framework. We roughly divide processors into two categories, client and server, and provide a lightweight configuration for clients.

## Get Started
### Get Default Algorithm Configuration

Users can get the default algorithm configuration by passing the processor_type explicitly to the get default configuration API, or leave it empty, and we will return the appropriate configuration according to the hardware information. Currently, the machine is detected as a server if one of the following conditions is met:

- If there is more than one sockets
- If the brand name includes `Xeon`
- If the DRAM size is greater than 32GB


> The last condition may not be very accurate, but models greater than 7B generally need more than 32GB, and we assume that the user won't try these models on a client machine.

Below is an example to get the default configuration of RTN.

```python
config_by_auto_detect = get_default_rtn_config()
config_for_client = get_default_rtn_config(processor_type="client")
config_for_server = get_default_rtn_config(processor_type="server")
```

### Set Environment Variables for Optimal Performance

To achieve optimal performance, we need to set the right environment variables. For example, [Intel® Core™ Ultra 7 Processor 155H](https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html) includes 6 P-cores and 10 E-cores. Use `taskset` to bind tasks on all P-cores to achieve optimal performance.

```bash
OMP_NUM_THREADS=12 taskset -c 0-11 python ./main.py
```

> Note: To detect the E-cores and P-cores on a Linux system, please refer [this](https://stackoverflow.com/a/71282744/23445462).

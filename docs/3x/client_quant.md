Quantization on Client

==========================================

1. [Introduction](#introduction)
2. [Support matrix](#supported-matrix)
3. [Get Started](#get-started) \
    2.1 [Get default lightweight algorithm configuration for client]\
    2.2 [Override the auto-detect result]\
    2.3 [Set several environment variables for optimal performance]
    
## Introduction
Currently, we supported different default algorithm configuration based on the type of machine for RTN, GPTQ, and Auto-Round on Pytorch framework.

## Support matrix


## Get Started
### Get default algorithm configuration

Currently, we detect the machine as server if one of below conditions meet, user can override it by setting the `processor_type` explicitly.

```python
config_for_client = get_default_rtn_config(processor_type="client")
```
### Compare the default configuration between client and server


### Set several environment variables for optimal performance
Takes [Intel® Core™ Ultra 7 Processor 155H](https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html) as example, it include 6 P-cores and 10 E-cores. Use `taskset` to bind task on all P-cores to achieve optimal performance.

```bash
taskset -c 0-11 python ./main.py
```

> Note: To detect the E-cores and P-cores in Linux system, please refer [here](https://stackoverflow.com/a/71282744/23445462).

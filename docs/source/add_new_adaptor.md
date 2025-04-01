

How to Add An Adaptor
=======


- [Introduction](#introduction)
- [API List that Need to Implement](#api-list-that-need-to-implement)
- [Design the framework YAML](#design-the-framework-yaml)
- [Add query_fw_capability API to Adaptor](#add-query-fw-capability-api-to-adaptor)
- [Add quantize API according to tune cfg](#add-quantize-api-according-to-tune-cfg)

## Introduction
Intel® Neural Compressor builds the low-precision inference solution on popular deep learning frameworks such as TensorFlow, PyTorch, Keras and ONNX Runtime. The adaptor layer is the bridge between the tuning strategy and vanilla framework quantization APIs, each framework has own adaptor. The users can add new adaptor to set strategy capabilities.

The document outlines the process of adding support for a new adaptor, in Intel® Neural Compressor with minimal changes. It provides instructions and code examples for implementation of a new adaptor. By following the steps outlined in the document, users can extend Intel® Neural Compressor's functionality to accommodate new adaptor and incorporate it into quantization workflows.

The quantizable operator behavior and it's tuning scope is defined in specific framework YAML file. The adaptor will parse this file and give the quantization capability to the Strategy object. Then Strategy will build tuning space of the specific graph/model and generate different tuning configuration from the tuning space to adaptor.

The diagram below illustrates all the relevant steps of how adaptor is invoked, with additional details provided for each annotated step.
```mermaid
  sequenceDiagram
  	autonumber
  	autonumber
    Adaptor ->> Adaptor: Design the framework YAML
    Strategy ->> Adaptor: use query_framework_capability to get the capability of the model
    Adaptor ->> Adaptor: Parse the framework YAML and get quantization capability
    Adaptor ->> Strategy: Send the capability including 'opwise' and 'optypewise' ability
    Strategy ->> Strategy: Build tuning space
    loop Traverse tuning space
    	Strategy->> Adaptor: generate next tuning cfg
        Adaptor ->> Adaptor: calibrate and quantize model based on tuning config

    end
```
❶ Design the framework YAML, inherit QueryBackendCapability class to parse the framework yaml.  
❷ Utilizes adaptor.query_fw_capability to query the framework's capabilities.  
❸ Parse the framework YAML and get quantization capability.  
❹ Send the capability including 'opwise' and 'optypewise' ability to Strategy.  
❺ Build the tuning space in Strategy.  
❻ Generates the tuning configurations for each operators of the model using the tuning space constructed in the previous step, specifying the desired tuning process.  
❼ Invokes the specific kernels for the calibration and quantization based on the tuning configuration.  

## API List that Need to Implement
These APIs are necessary to add a new adapter. Here are the parameter types and functionality descriptions of these APIs. The following chapters will introduce the specific implementation and data format of these APIs in detail.

| API | Parameters |Output   |Usage   | Comments|
| :------ | :------|:------ |:------ | :------ |
| query_fw_capability(self, model) | **model** (object): A INC model object to query quantization tuning capability. |output format: <br> {'opwise': {(node_name, node_op): [{'weight': {'dtype': ...#int8/fp32 or other data type}, 'activation': {'dtype': ...#int8/fp32 or other data type}}, ...]},<br> 'optypewise':{node_op: [{'weight': {'dtype': ...#int8/fp32 or other data type}, 'activation': {'dtype': ...#int8/fp32}}], ...}} |The function is used to return framework tuning capability. |Confirm the the data format output by the function must meet the requirements |
| quantize(self, tune_cfg, model, dataloader, q_func=None) | **tune_cfg** (dict): the chosen tuning configuration.<br> **model** (object): The model to do quantization.<br>**dataloader** (object): The dataloader used to load quantization dataset. **q_func**(optional): training function for quantization aware training mode.| output quantized model object|This function use the dataloader to generate the data required by the model, and then insert Quantize/Dequantize operator into the quantizable op required in the tune_config and generate the model for calibration, after calibration, generate the final quantized model according to the obtained data range from calibration| |

## Design the framework YAML
To enable accuracy-aware tuning with a specific framework, we should define the [framework YAML](./framework_yaml.md) which unifies the configuration format for quantization and provides a description for the capabilities of the specific framework. 
>**Note**: You should refer to [framework_yaml.md](./framework_yaml.md) to define the framework specific YAML.

## Add query_fw_capability to Adaptor
Each framework adaptor should implement the `query_fw_capability` function, this function will only be invoked once and will loop over the graph/model for the quantizable operators and collect each operator's opwise details and optypewise capability. You should return a standard dict of the input model's tuning capability. The format is like below:

```python
capability = {
    'opwise': {('conv2d', 'Conv2D'): [int8_conv_config, {'weight': {'dtype': 'bf16'}, 'activation': {'dtype': 'bf16'}}, {'weight': {'dtype': 'fp32'}, 'activation': {'dtype': 'fp32'}}], ... }# all quantizable opwise key-value pair with key tuple: (node_name, node_op)}
    'optypewise': optype_wise_ability,
}
```
The int8_conv_config is like below, it's parsed from the framework YAML.
```python
int8_conv_config = {
    "weight": {
        "dtype": "int8",
        "algorithm": "minmax",
        "granularity": "per_channel",
        "scheme": "sym",
    },
    "activation": {
        "dtype": "int8",
        "quant_mode": "static",
        "algorithm": "kl",
        "granularity": "per_tensor",
        "scheme": "sym",
    },
}
```
The `optype_wise_ability` example config is like below.

```python
optype_wise_ability = {
    'Conv2D': {
        'weight': {
               'dtype': 'int8',
               'algorithm': 'minmax',
               'granularity': 'per_channel',
               'scheme': 'sym',
        },
        'activation': {
               'dtype': 'int8',
               'quant_mode': 'static',
               'algorithm': 'kl',
               'granularity': 'per_tensor',
               'scheme': 'sym',
        },
    },
    ... #all optype wise ability
}
```
After the work above, we have implement the `query_fw_capability` API and get the tuning capability dict for the Strategy object. Then the Strategy object will fetch tuning configuration and give to the quantize API to get the quantized model.

## Add quantize API according to tune_cfg
`quantize` function is used to perform quantization for post-training quantization and quantization-aware training. Quantization processing includes calibration and conversion processing for post-training quantization, while for quantization-aware training, it includes training and conversion processing.

The first work of `quantize` function is to invoke `tuning_cfg_to_fw` to generate the self.quantize_config. The self.quantize_config is a dict including the quantization information. Its format is like below

```
self.quantize_config  = {
    'device': 'CPU',
    'calib_iteration': 50,
    'advance': {},
    'op_wise_config': {
        'conv2d': (True, 'minmax', False, 8),
        # ....
    }
}
```
As the Strategy object will decide which operator to quantize or not, some quantizable operators may not be in the `tune_cfg`. Only dispatched operators will be set to the `op_wise_config` in `self.quantize_config`. `op_wise_config` is a dict with format like 

```
op_wise_config = {
    op_name: (is_perchannel, 
              algorithm, 
              is_asymmetric, 
              weight_bit)
} 
```

You can also set bf16_ops in `tuning_cfg_to_fw` and the `self.bf16_ops` will be converted in `convert_bf16` function.

After got the `self.quantize_config`, we can prepare to quantize the model. It usually have three steps.

### Prepare calibration model from fp32 graph
The calibration process needs to collect the activation and weight during inference. After collection, a reasonable data range is calculated for subsequent data type conversion. You should to prepare the calibration model in the quantize API.

### Run sampling iterations of the fp32 graph to calibrate quantizable operators.
When we get the calib_model, We should run calibration on this model and collect the fp32 activation data. In this step, we will use the dataloader and forward the model. 

### Calculate the data range and generate quantized model
Calibration data can only approximate the data distribution of the entire dataset, larger sampling size means a more complete approximation of the data distribution, but it will also introduce some outliers, which will cause the data range obtained to be somewhat distorted.

 You can use different algorithms to make the data range more in line with the real data distribution. After applying these algorithms, we obtained the data distribution range of each operator. At this time, you can generate the quantized model.

This quantized model can be evaluated. If the evaluation meets the set metric goal, the entire quantization process will be over. Otherwise, a new tuning configuration will be generated until a quantized model that meets the metric requirements.

Framework YAML Configuration Files
====
1. [Introduction](#introduction)
2. [Supported Feature Matrix](#supported-feature-matrix)
2. [Get Started with Framework YAML Files](#get-started-with-framework-yaml-files)



## Introduction

Intel® Neural Compressor uses YAML files for quick 
and user-friendly configurations. There are two types of YAML files - 
user YAML files and framework YAML files, which are used in 
running user cases and setting up framework capabilities, respectively. 

Here, we introduce the framework YAML file, which describes the behavior of 
a specific framework. There is a corresponding framework YAML file for each framework supported by 
Intel® Neural Compressor - TensorFlow
, Intel® Extension for TensorFlow*, PyTorch, Intel® Extension for PyTorch*, ONNX Runtime, and MXNet. 

>**Note**: Before diving to the details, we recommend that the end users do NOT make modifications
unless they have clear requirements that can only be met by modifying the attributes. 

## Supported Feature Matrix

| Framework  | YAML Configuration Files |
|------------|:------------------------:|
| TensorFlow |         &#10004;         |
| PyTorch    |         &#10004;         |
| ONNX       |         &#10004;         |
| MXNet      |         &#10004;         |


## Get started with Framework YAML Files

For the purpose of framework setup, let's take a look at a tensorflow framework YAML file;
other framework YAML files follow same syntax. A framework YAML file specifies following
information and capabilities for current runtime framework. Let's go through 
them one by one: 

* ***version***: This specifies the supported versions. 
```yaml
  version:
    name: ['2.1.0', '2.2.0', '2.3.0', '2.4.0', '2.5.0', '2.6.0', '2.6.1', '2.6.2', '2.7.0', '2.8.0', '1.15.0-up1', '1.15.0-up2']
```

* ***precisions***: This defines the supported precisions of specific versions. 
```yaml
  precisions: 
    names: int8, uint8, bf16, fp32
    valid_mixed_precisions: []
```
* ***op***: This defines a list of valid OP types for each precision.
```yaml
  ops: 
    int8: ['Conv2D', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool']
    uint8: ['Conv2D', 'DepthwiseConv2dNative', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool']
    bf16: ['Conv2D']  
    fp32: ['*'] # '*' means all op types
```
* ***capabilities***: This defines the quantization ability of specific ops, such as
granularity, scheme, and algorithm. The activation assumes that input and output activations
share the same data type by default, which is based on op semantics defined by
frameworks. 
```yaml
  capabilities: 
    int8: {
          'Conv2D': {
            'weight': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_channel','per_tensor'],
                        'algorithm': ['minmax']
                        },
            'activation': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax', 'kl']
                        }
                    },
          'MatMul': {
            'weight': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax']
                        },
            'activation': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['asym', 'sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax']
                        }
                    },
          'default': {
            'activation': {
                        'dtype': ['uint8', 'fp32'],
                        'algorithm': ['minmax'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor']
                        }
                    },
          }

    uint8: {
          'Conv2D': {
            'weight': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_channel','per_tensor'],
                        'algorithm': ['minmax']
                        },
            'activation': {
                        'dtype': ['uint8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax', 'kl']
                        }
                    },
          'MatMul': {
            'weight': {
                        'dtype': ['int8', 'fp32'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax']
                        },
            'activation': {
                        'dtype': ['uint8', 'fp32'],
                        'scheme': ['asym', 'sym'],
                        'granularity': ['per_tensor'],
                        'algorithm': ['minmax']
                        }
                    },
          'default': {
            'activation': {
                        'dtype': ['uint8', 'fp32'],
                        'algorithm': ['minmax'],
                        'scheme': ['sym'],
                        'granularity': ['per_tensor']
                        }
                    },
          }
```
* ***patterns***: This defines the supported fusion sequence for each op. 
```yaml
  patterns: 
    fp32: [ 
        'Conv2D + Add + Relu',
        'Conv2D + Add + Relu6',
        'Conv2D + Relu',
        'Conv2D + Relu6',
        'Conv2D + BiasAdd'
        ]
    int8: [
        'Conv2D + BiasAdd',
        'Conv2D + BiasAdd + Relu',
        'Conv2D + BiasAdd + Relu6'
        ]
    uint8: [
        'Conv2D + BiasAdd + AddN + Relu',
        'Conv2D + BiasAdd + AddN + Relu6',
        'Conv2D + BiasAdd + AddV2 + Relu',
        'Conv2D + BiasAdd + AddV2 + Relu6',
        'Conv2D + BiasAdd + Add + Relu',
        'Conv2D + BiasAdd + Add + Relu6',
        'Conv2D + BiasAdd + Relu',
        'Conv2D + BiasAdd + Relu6',
        'Conv2D + Add + Relu',
        'Conv2D + Add + Relu6',
        'Conv2D + Relu',
        'Conv2D + Relu6',
        'Conv2D + BiasAdd',
        'DepthwiseConv2dNative + BiasAdd + Relu6',
        'DepthwiseConv2dNative + BiasAdd + Relu',
        'DepthwiseConv2dNative + Add + Relu6',
        'DepthwiseConv2dNative + BiasAdd',
        'MatMul + BiasAdd + Relu',
        'MatMul + BiasAdd',
  ]
```

* ***grappler_optimization***: This defines the grappler optimization. 
```yaml
  grappler_optimization: 
    pruning: True                                    # optional. grappler pruning optimizer,default value is True.
    shape: True                                      # optional. grappler shape optimizer,default value is True.
    constfold: False                                 # optional. grappler constant folding optimizer, default value is True.
    arithmetic: False                                # optional. grappler arithmetic optimizer,default value is False.
    dependency: True                                 # optional. grappler dependency optimizer,default value is True.
    debug_stripper: True                             # optional. grappler debug_stripper optimizer,default value is True.
    loop: True                                       # optional. grappler loop optimizer,default value is True.

```

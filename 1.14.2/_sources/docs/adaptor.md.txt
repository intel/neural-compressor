Adaptor
=======

## Introduction

Intel® Neural Compressor builds the low-precision inference
solution on popular deep learning frameworks such as TensorFlow, PyTorch,
MXNet, and ONNX Runtime. The adaptor layer is the bridge between the 
tuning strategy and vanilla framework quantization APIs.

## Adaptor Design

Neural Compressor supports a new adaptor extension by
implementing a subclass `Adaptor` class in the neural_compressor.adaptor package
and registering this strategy by the `adaptor_registry` decorator.

For example, a user can implement an `Abc` adaptor like below:

```python
@adaptor_registry
class AbcAdaptor(Adaptor):
    def __init__(self, framework_specific_info):
        ...

    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        ...

    def evaluate(self, model, dataloader, postprocess=None,
                 metric=None, measurer=None, iteration=-1, tensorboard=False):
        ...

    def query_fw_capability(self, model):
        ...

    def query_fused_patterns(self, model):
        ...
```

* `quantize` function is used to perform calibration and quantization in post-training quantization.
* `evaluate` function is used to run an evaluation on a validation dataset.
* `query_fw_capability` function is used to run a query framework quantization capability and intersects with the user yaml configuration.
* `query_fused_patterns` function is used to run a query framework graph fusion capability and decide the fusion tuning space.

### Query API

#### Background

Besides the adaptor API, we also introduced the Query API which describes the
behavior of a specific framework. With this API, Neural Compressor can easily query the
following information on the current runtime framework.

*  The runtime version information.
*  The Quantizable ops type.
*  The supported sequence of each quantizable op.
*  The instance of each sequence.

In the past, the above information was generally defined and hidden in every corner of the code which made effective maintenance difficult. With the Query API, we only need to create one unified yaml file and call the corresponding API to get the information. For example, the [tensorflow.yaml](../neural_compressor/adaptor/tensorflow.yaml) keeps the current Tensorflow framework ability. We recommend that the end user not make modifications if requirements are not clear.

#### Unify Config Introduction

Below is a fragment of the Tensorflow configuration file.

* **precisions** field defines the supported precision for Neural Compressor.
    -  valid_mixed_precision enumerates all supported precision combinations for specific scenario. For example, if one hardware doesn't support bf16， it should be `int8 + fp32`.
* **ops** field defines the valid OP type list for each precision.
* **capabilities** field focuses on the quantization ability of specific ops such as granularity, scheme, and algorithm. The activation assumes the same data type for both input and output activation by default based on op semantics defined by frameworks.
* **patterns** field defines the supported fusion sequence of each op.

```yaml
---
-
  version:
    name: '2.4.0'
  
  precisions: &common_precisions
    names: int8, uint8, bf16, fp32
    valid_mixed_precisions: []
  
  ops: &common_ops
    int8: ['Conv2D', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool']
    uint8: ['Conv2D', 'DepthwiseConv2dNative', 'MatMul', 'ConcatV2', 'MaxPool', 'AvgPool']
    bf16: ['Conv2D']  #TODO need to add more bf16 op types here
    fp32: ['*'] # '*' means all op types
  
  capabilities: &common_capabilities
    int8: &ref_2_4_int8 {
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
                        'algorithm': ['minmax', 'kl']
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

    uint8: &ref_2_4_uint8 {
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
                        'algorithm': ['minmax', 'kl']
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

  patterns: &common_patterns
    fp32: [ #TODO Add more patterns here to demonstrate our concept the results external engine should return.
        'Conv2D + Add + Relu',
        'Conv2D + Add + Relu6',
        'Conv2D + Relu',
        'Conv2D + Relu6',
        'Conv2D + BiasAdd'
        ]
    int8: ['Conv2D + BiasAdd', 'Conv2D + BiasAdd + Relu', 'Conv2D + BiasAdd + Relu6']
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
        'DepthwiseConv2dNative + Add + Relu6',
        'DepthwiseConv2dNative + BiasAdd',
        'MatMul + BiasAdd + Relu',
        'MatMul + BiasAdd',
  ]
```
#### Query API Introduction

The abstract class `QueryBackendCapability` is defined in [query.py](../neural_compressor/adaptor/query.py#L21). Each framework should inherit it and implement the member function if needed. Refer to Tensorflow implementation [TensorflowQuery](../neural_compressor/adaptor/tensorflow.py#L628).


## Customize a New Framework Backend

Look at onnxruntime as an example. ONNX Runtime is a backend proposed by Microsoft, and is based on the MLAS kernel by default.
Onnxruntime already has [quantization tools](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization), so the question becomes how to integrate onnxruntime quantization tools into Neural Compressor.

1. Capability
   
   The user should explore quantization capability first. According to [onnx_quantizer](https://github.com/microsoft/onnxruntime/blob/503b61d897074a494f5798069308ee67d8fb9ace/onnxruntime/python/tools/quantization/onnx_quantizer.py#L76), the quantization tools support the following attributes:
   * whether per_channel
   * whether reduce_range
   * QLinear mode or Integer mode (which is only seen in onnxruntime)
   * whether static (static quantization or dynamic quantization)
   * weight_qtype (choices are float32, int8 and uint8)
   * input_qtype (choices are float32, int8 and uint8)
   * quantization_params (None if dynamic quantization)
   * &1.8 nodes_to_quantize, nodes_to_exclude
   * op_types_to_quantize

   We can pass a tune capability to Neural Compressor such as:

   ```yaml
   {'optypewise': {'conv': 
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']},
                    'algorithm': ['minmax', ],
                    'granularity': ['per_channel']
                   }, 
                   'matmul': 
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']},
                    'algorithm': ['minmax', ],
                    'granularity': ['per_channel']
                   }
                   }, 
    'opwise':  {('conv1', 'conv'):
                   {
                    'activation': { 'dtype': ['uint8', 'fp32']},
                    'weight': {'dtype': ['int8', 'fp32']}
                   }
                   }
    }
   ```

2. Parse tune config
   
   Neural Compressor can generate a tune config from your tune capability such as the
   following: 

   ```yaml
    {
        'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']],
        'fp32': [['CONV2D', 'RELU', 'BN']]}, 
        'calib_iteration': 10,
        'op': {
        ['op1', 'CONV2D']: {
            'activation':  {'dtype': 'uint8',
                            'algorithm': 'minmax',
                            'scheme':'sym',
                            'granularity': 'per_tensor'},
            'weight': {'dtype': 'int8',
                        'algorithm': 'kl',
                        'scheme':'asym',
                        'granularity': 'per_channel'}
        },
        ['op2', 'RELU]: {
            'activation': {'dtype': 'int8',
                            'scheme': 'asym',
                            'granularity': 'per_tensor',
                            'algorithm': 'minmax'}
        },
        ['op3', 'CONV2D']: {
            'activation':  {'dtype': 'fp32'},
            'weight': {'dtype': 'fp32'}
        },
        ...
        }
    }
   ```
   Then you can parse this config into a format that ONNXQuantizer can accept.
   Verify whether your quantization API supports model wise or op wise quantization. For example, node "conv1" uses the "minmax" algorithm and node "conv2" uses the "KL" algorithm, or the whole model must use "minmax" or "KL" in general.

3. Pre-optimize
   If your backend supports FP32 graph optimization, you can apply it in **query_fw_capability** and quantize your optimized fp32 model instead of
   the original model: 
   >model = self.pre_optimized_model if self.pre_optimized_model else model

4. Do quantization
   
   This part depends on your backend implementations. Refer to [onnxruntime](../neural_compressor/adaptor/onnxrt.py) as an example.

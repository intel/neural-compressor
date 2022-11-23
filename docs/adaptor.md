Adaptor
=======
1. [Adaptor Layer Introduction](#adaptor-layer-introduction)
2. [Working Flow](#working-flow)
3. [Adaptor API Summary](#adaptor-api-summary)

    3.1 [Query API](#query-api)

4. [Example of adding a new backend support](#example-of-adding-a-new-backend-support)

    4.1 [Capability](#capability)

    4.2 [Implement ONNXRTAdaptor Class](#implement-onnxrtadaptor-class)

    4.3 [Pre-optimize](#pre-optimize)

    4.4 [Setting Tune Config](#setting-tune-config)

    4.5 [Do Quantization](#do-quantization)

## Adaptor Layer Introduction

Intel® Neural Compressor builds the low-precision inference
solution on popular deep learning frameworks such as TensorFlow, PyTorch,
MXNet, and ONNX Runtime. The adaptor layer is the bridge between the 
tuning strategy and vanilla framework quantization APIs.

## Working Flow
Adaptor only provide framework API for tuning strategy. So we can find complete working flow in [tuning strategy working flow](./tuning_strategies.md).

## Adaptor API Summary

Neural Compressor supports a new adaptor extension by
implementing a subclass `Adaptor` class in the neural_compressor.adaptor package
and registering this adaptor by the `adaptor_registry` decorator.

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

* `quantize` function is used to perform quantization for post-training quantization and quantization-aware training. Quantization processing includes calibration and conversion processing for post-training quantization, while for quantization-aware training, it includes training and conversion processing.
* `evaluate` function is used to run an evaluation on a validation dataset. It is a built-in function, if user wants to use specifical evaluation function, he can pass the evaluation function to quantizer.
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


## Example of Adding a New Backend Support

Look at onnxruntime as an example. ONNX Runtime is a backend proposed by Microsoft, and is based on the MLAS kernel by default.
Onnxruntime already has [quantization tools](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization), so the question becomes how to integrate onnxruntime quantization tools into Neural Compressor.

### Capability
   
   The user should explore quantization capability first. According to [onnx_quantizer](https://github.com/microsoft/onnxruntime/blob/503b61d897074a494f5798069308ee67d8fb9ace/onnxruntime/python/tools/quantization/onnx_quantizer.py#L76), the quantization tools support the following attributes:
   * whether per_channel
   * whether reduce_range
   * QLinear mode, QDQ mode or Integer mode (which is only seen in onnxruntime)
   * whether static (static quantization or dynamic quantization)
   * weight_qtype (choices are float32, int8 and uint8)
   * input_qtype (choices are float32, int8 and uint8)
   * quantization_params (None if dynamic quantization)
   * nodes_to_quantize, nodes_to_exclude
   * op_types_to_quantize

   We define three configuration files to describe the capability of ONNXRT. Please refer to [onnxrt_qlinear.yaml](../neural_compressor/adaptor/onnxrt_qlinear.yaml), [onnxrt_integer.yaml](../neural_compressor/adaptor/onnxrt_integer.yaml) and [onnxrt_qdq.yaml](../neural_compressor/adaptor/onnxrt_qdq.yaml).

   ```yaml  # qlinear
    version:
      name: '1.6.0'

    precisions: &common_precisions
      names: int8, uint8, fp32
      valid_mixed_precisions: []

    ops:
      int8: ['Conv', 'MatMul', 'Attention', 'Mul', 'Relu', 'Clip', 
          'LeakyRelu', 'Gather', 'Sigmoid', 'MaxPool', 'EmbedLayerNormalization',
          'FusedConv', 'GlobalAveragePool', 'Add']     
      fp32: ['*'] # '*' means all op types

    capabilities: &common_capabilities
      int8: &ref_1_6 {
            'FusedConv': &key_1_6_0 {
              'weight': {
                          'dtype': ['int8'],
                          'scheme': ['sym'],
                          'granularity': ['per_channel', 'per_tensor'],
                          'algorithm': ['minmax']
                          },
              'activation': {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'granularity': ['per_tensor'],
                          'algorithm': ['minmax']
                          }
                      },
            'Conv': {
              'weight':   {
                          'dtype': ['int8'],
                          'scheme': ['sym'],
                          'granularity': ['per_channel', 'per_tensor'],
                          'algorithm': ['minmax']
                          },
              'activation': {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'granularity': ['per_tensor'],
                          'algorithm': ['minmax']
                          }
                      },
            'Gather': {
              'weight':   {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'algorithm': ['minmax'],
                          'granularity': ['per_channel', 'per_tensor'],
                          },
              'activation': {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'algorithm': ['minmax'],
                          'granularity': ['per_tensor'],
                          }
                      },
            'MatMul': {
              'weight':   {
                          'dtype': ['int8'],
                          'scheme': ['sym'],
                          'granularity': ['per_tensor'],
                          'algorithm': ['minmax']
                          },
              'activation': {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'granularity': ['per_tensor'],
                          'algorithm': ['minmax']
                          }
                      },
            'default': {
               'weight': {
                          'dtype': ['int8'],
                          'scheme': ['sym'],
                          'algorithm': ['minmax'],
                          'granularity': ['per_tensor']
                      },
               'activation': {
                          'dtype': ['uint8'],
                          'scheme': ['asym'],
                          'algorithm': ['minmax'],
                          'granularity': ['per_tensor']
                          }
                      },
            }

    graph_optimization: &default_optimization  # from onnxruntime graph_optimization_level
        level: 'ENABLE_EXTENDED'         # choices are ['DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL']
   ```

### Implement ONNXRTAdaptor Class

   The base class ONNXRTAdaptor inherits from the Adaptor class. Please refer to [onnxrt.py](../neural_compressor/adaptor/onnxrt.py).

   ```python
    @adaptor_registry
    class ONNXRT_QLinearOpsAdaptor(ONNXRTAdaptor):
      @dump_elapsed_time("Pass quantize model")
      def quantize(self, tune_cfg, model, data_loader, q_func=None):
        ......

      @dump_elapsed_time("Pass recover model")
      def recover(self, model, q_config):
        ......

      def inspect_tensor(self, model, dataloader, op_list=[],
                       iteration_list=[],
                       inspect_type='activation',
                       save_to_disk=False,
                       save_path=None,
                       quantization_cfg=None):
        ......

      def set_tensor(self, model, tensor_dict):
        ......

      def query_fw_capability(self, model):
        ......

      def evaluate(self, input_graph, dataloader, postprocess=None,
                 metrics=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        ......

      def diagnosis_helper(self, fp32_model, int8_model, tune_cfg=None, save_path=None):
        ......

      def save(self, model, path):
        ......
   ```

### Pre-optimize

   If your backend supports FP32 graph optimization, you can apply it in **query_fw_capability** and quantize your optimized fp32 model instead of
   the original model: 
   >model = self.pre_optimized_model if self.pre_optimized_model else model

### Setting Tune Config

   Now we can use onnxrt adaptor to compress the onnx model by setting tune config.

   Tuning config is a yaml file to control the behavior of compressor. please refer to [example](../examples/onnxrt/image_recognition/mobilenet_v2/quantization/ptq/mobilenet_v2.yaml).

   ```yaml

   model:
      name: xxx                             # mandatory. the model name.
      framework: onnxrt_qlinearops

   device: cpu

   tuning:
      strategy:
        name: basic
      accuracy_criterion:
        relative:  0.01
      objective: performance
      exit_policy:
        timeout: 0
        max_trials: 100
      ......

   quantization:
      approach: post_training_static_quant
      op_wise: {
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
   ```
   Then you can parse this config into a format that ONNXQuantizer can accept.
   Verify whether your quantization API supports model wise or op wise quantization. For example, node "conv1" uses the "minmax" algorithm and node "conv2" uses the "KL" algorithm, or the whole model must use "minmax" or "KL" in general.

### Do Quantization

   Refer to [example](../examples/onnxrt/image_recognition/mobilenet_v2/quantization/ptq/README.md) as an example.
Adaptor
=======
1. [Introduction](#introduction)
2. [Adaptor Support Matrix](#adaptor-support-matrix)
3. [Working Flow](#working-flow)
4. [Get Started with Adaptor API](#get-start-with-adaptor-api)

    4.1 [Query API](#query-api)

5. [Example of adding a new backend support](#example-of-adding-a-new-backend-support)

    5.1 [Capability](#capability)

    5.2 [Implement ONNXRTAdaptor Class](#implement-onnxrtadaptor-class)

## Introduction

Intel® Neural Compressor builds the low-precision inference
solution on popular deep learning frameworks such as TensorFlow, PyTorch,
MXNet, and ONNX Runtime. The adaptor layer is the bridge between the 
tuning strategy and vanilla framework quantization APIs.

## Adaptor Support Matrix

|Framework     |Adaptor      |
|--------------|:-----------:|
|TensorFlow    |&#10004;     |
|PyTorch       |&#10004;     |
|ONNX          |&#10004;     |
|MXNet         |&#10004;     |


## Working Flow
Adaptor only provide framework API for tuning strategy. So we can find complete working flow in [tuning strategy working flow](./tuning_strategies.md).

## Get Started with Adaptor API

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
* `evaluate` function is used to run an evaluation on a validation dataset. It is a built-in function, if user wants to use specific evaluation function, he can pass the evaluation function to quantizer.
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

Below is a fragment of the Tensorflow configuration file.

* **precisions** field defines the supported precision for Neural Compressor.
    -  valid_mixed_precision enumerates all supported precision combinations for specific scenario. For example, if one hardware doesn't support bf16， it should be `int8 + fp32`.
* **ops** field defines the valid OP type list for each precision.
* **capabilities** field focuses on the quantization ability of specific ops such as granularity, scheme, and algorithm. The activation assumes the same data type for both input and output activation by default based on op semantics defined by frameworks.
* **patterns** field defines the supported fusion sequence of each op.

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

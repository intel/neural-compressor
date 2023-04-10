

New Data Type
=======


 1. [Introduction](#introduction)
 2. [Defines the Quantization Ability of the Specific Operator](#defines-the-quantization-ability-of-the-specific-operator)
 3. [Use the New Data Type](#use-the-new-data-type)
 4. [Others](#others)

## Introduction
Deep Learning frameworks like PyTorch, ONNX Runtime, and TensorFlow currently do not natively support quantization with precision lower than 8-bit. However, it is possible to simulate lower-precision quantization by specifying the value ranges of a given data type. INC provides flexibility for users to extend its functionality by adding new data types to the framework.

This document provides guidance on how to add new data types to the INC by using the example of extending the PyTorch `Conv2d` operator to support 4-bit quantization.

## Defines the Quantization Ability of the Specific Operator

The first step in adding a new data type to INC is to define the capabilities of the new data type itself and include it to the framework YAML. 
The capabilities should include the quantized data types and quantization schemes of activation and weight(optional) respectively. The following table descript the detail of each filed:


| Field name | Options | Description |
| -----------|---------------|------------
| Data Type (`dtype`) | `uint4`, `int4` | The quantization data type being added. It use 4-bit as example, where `uint4` represents an unsigned 4-bit integer and `int4` represents a signed 4-bit integer.|
| Quantization (`scheme`) | `sym`, `asym`| The quantization scheme used for the new data type. `sym` represents symmetric quantization, `asym` represents asymmetric quantization.|
| Quantization Granularity (`granularity`)| `per_channel`, `per_tensor`| The granularity at which quantization is applied. `per_channel` represents that the quantization is applied independently per channel, `per_tensor` represents that the quantization is applied to the entire tensor as a whole. |
| Calibration Algorithm (`algorithm`)| `minmax`, `kl`| 	The calibration algorithm used for the new data type. `minmax` represents the minimum-maximum algorithm, `kl` represents the Kullback-Leibler divergence algorithm. |


For example, let's add  4-bit quantization for `Conv2d` in the PyTorch backend. We can modify the `neural_compressor/adaptor/pytorch_cpu.yaml` as follows:

```diff
  ...
  fp32: ['*'] # `*` means all op types.
  

+    int4: {
+        'static': {
+            'Conv2d': {
+                'weight': {
+                    'dtype': ['int4'],
+                    'scheme': ['sym'],
+                    'granularity': ['per_channel'],
+                    'algorithm': ['minmax']},
+                'activation': {
+                    'dtype': ['uint4'],
+                    'scheme': ['sym'],
+                    'granularity': ['per_tensor'],
+                    'algorithm': ['minmax']},
+            },
+        }
+    }

  int8: &1_11_capabilities {
    'static': &cap_s8_1_11 {
          'Conv1d': &cap_s8_1_11_Conv1d {
  ...

```
The code states that the PyTorch Conv2d Operator has the ability to quantize weights to int4 using the `torch.per_channel_symmetric` quantization scheme, , with the supported calibration algorithm being `minmax`. Additionally, the operator can quantize activations to `uint4` using the `torch.per_tensor_symmetric` quantization scheme, with the supported calibration algorithm also being `minmax`.

> Note: more details about the framework YAML can be found [here](./framework_yaml.md).


### Use the New Data Type

Once the new data type has been added to INC, it can be used in the same way as any other data type within the framework. To specify that all `Conv2d` operators should utilize 4-bit quantization, the following steps can be taken:

```python
from neural_compressor.config import PostTrainingQuantConfig
op_type_dict = {
    'Conv2d': {
        'weight': {
            'dtype': ['int4']
        },
        'activation': {
            'dtype': ['uint4']
        }
    }
}
conf = PostTrainingQuantConfig(op_type_dict=op_type_dict)
...

```

This code specifies quantization rules for all `Conv2d` operators, quantizing their weight with `int4` and their activation with `uint4`.


### Others
It is worth noting that currently, INC only supports the PyTorch backend with N-bit quantization, where N is an integer between 1 and 7. Other backends or quantization schemes may be added in the future.

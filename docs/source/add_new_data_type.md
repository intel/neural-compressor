

New Data Type
=======


 1. [Introduction](#introduction)
 2. [Defines the Quantization Ability of the Specific Operator](#defines-the-quantization-ability-of-the-specific-operator)
 3. [Use the New Data Type](#use-the-new-data-type)
 4. [Others](#others)

## Introduction
Currently, deep Learning frameworks such as PyTorch, ONNX Runtime, and Tensorflow do not have native support for quantization with lower than 8-bit precision. However, it is possible to simulate lower-precision quantization by specifying the value ranges of a given data type. INC provides flexibility that allows users to extend its functionality by adding these new data types.

This document provides guidance on how to add new data types to the INC, using the example of extending the PyTorch `Conv2d` operator to support 4-bit quantization.

## Defines the Quantization Ability of the Specific Operator

The first step in adding a new data type to INC is to define the capabilities of the new data type itself and include it to the framework YAML. 
The capabilities include the quantized data types and quantization schemes of activation and weight(optional) respectively.


| Field name | Options | Description |
| -----------|---------------|------------
| `dtype` | `uint4`, `int4` | Quantization data type |
| `scheme` | `sym`, `asym`| Quantization scheme |
|`granularity`| `per_channel`, `per_tensor`| Quantization granularity |
|`algorithm`| `minmax`, `kl`| Calibration algorithm |


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
The code states that the PyTorch Conv2d Operator has the ability to quantize weights to int4 using the `torch.per_channel_symmetric` quantization scheme, and the supported calibration algorithm for this is `minmax`. Additionally, the operator can quantize activations to `uint4` using the `torch.per_tensor_symmetric` quantization scheme, and the supported calibration algorithm for this is also `minmax`.

> Note: more details about the framework YAML can be found [here](./framework_yaml.md).


### Use the New Data Type

After adding the new data type to INC, it can be used in the same manner as any other data type within the INC. For instance, to specify that all `Conv2d` operators utilize 4-bit quantization, we can do it as follows:

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



### Support Matrix

| Field name | Description |
| -----------|---------------
| Support data types to extend | N-bit, N is an integer between 1 and 7, indicating the number of quantized bits|
| Support Framework | PyTorch|



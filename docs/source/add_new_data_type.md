

New Data Type
=======


 1. [Introduction](#introduction)
 2. [Defines the Quantization Ability of the Specific Operator](#defines-the-quantization-ability-of-the-specific-operator)
 3. [Use the New Data Type](#use-the-new-data-type)
 4. [Others](#others)

## Introduction
Currently, deep Learning frameworks such as PyTorch, ONNX, and Tensorflow do not have native support for quantization with lower than 8-bit, such as 4-bit. However, it is possible to simulate it by setting the value ranges of the data type. INC provides flexibility that allows users to extend its functionality by adding these new data types.

This document provides instructions for adding 4-bit quantization to the `Conv2d` operator in the INC PyTorch backend.

## Defines the Quantization Ability of the Specific Operator

The first step in adding a new data type to INC is to define the capabilities of the new data type itself add include it to the framwork YAML.  For example, let's add  4-bit quantization for `Conv2d` in the PyTorch backend. We can modify the `neural_compressor/adaptor/pytorch_cpu.yaml`  as follows:

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

This code specifies quantization rules for all `Conv2d` operators, quantizing their weight with `int4` and their activation with `uint8`.



## Others

We can extend the INC's functionality to support various lower bit quantizations such as 2-bit, 6-bit, and beyond.




How to Support New Data Type, Like Int4, with a Few Line Changes
=======


- [Introduction](#introduction)
- [Define the Quantization Ability of the Specific Operator](#define-the-quantization-ability-of-the-specific-operator)
- [Invoke the Operator Kernel According to the Tuning Configuration](#invoke-the-operator-kernel-according-to-the-tuning-configuration)
- [Use the New Data Type](#use-the-new-data-type)
- [Summary](#summary)

## Introduction
To enable accuracy-aware tuning with various frameworks, Intel® Neural Compressor introduced the [framework YAML](./framework_yaml.md) which unifies the configuration format for quantization and provides a description for the capabilities of specific framework. Before explaining how to add a new data type, let's first introduce the overall process, from defining the operator behavior in YAML to invoking it by the adaptor. The diagram below illustrates all the relevant steps, with additional details provided for each annotated step.

> Note: The `adaptor` is a layer that abstracts various frameworks supported by Intel® Neural Compressor.



```mermaid
  sequenceDiagram
  	autonumber
    Strategy ->> Adaptor: query framework capability
    Adaptor ->> Strategy: Parse the framework YAML and return capability
    Strategy ->> Strategy: Build tuning space
    loop Traverse tuning space
    	Strategy->> Adaptor: generate next tuning cfg
        Adaptor ->> Adaptor: calibrate and quantize model based on tuning config

    end
```
1. **Strategy**: Drives the overall tuning process and utilizes `adaptor.query_fw_capability` to query the framework's capabilities.

2. **Adaptor**: Parses the framework YAML, filters some corner cases, and constructs the framework capability. This includes the capabilities of each operator and other model-related information.

3. **Strategy**: Constructs the tuning space based on the framework capability and initiates the tuning process.

4. **Strategy**: Generates the tuning configurations for each operators of the model using the tuning space constructed in the previous step, specifying the desired tuning process.

5. **Adaptor**: Invokes the specific kernels for the calibration and quantization based on the tuning configuration.


The following section provides an example of extending the PyTorch `Conv2d` operator to include support for 4-bit quantization.

## Define the Quantization Ability of the Specific Operator

The first step in adding a new data type for specific operator to Intel® Neural Compressor is to extend the capabilities of operator and include it to the framework YAML.
The capabilities should include the quantized data types and quantization schemes of activation and weight(if applicable). The following table describes the detail of each filed:


| Field name | Options | Description |
| -----------|---------------|------------
| Data Type (`dtype`) | `uint4`, `int4` | The quantization data type being added. It use 4-bit as example, where `uint4` represents an unsigned 4-bit integer and `int4` represents a signed 4-bit integer.|
| Quantization (`scheme`) | `sym`, `asym`| The quantization scheme used for the new data type. `sym` represents symmetric quantization, `asym` represents asymmetric quantization.|
| Quantization Granularity (`granularity`)| `per_channel`, `per_tensor`| The granularity at which quantization is applied. `per_channel` represents that the quantization is applied independently per channel, `per_tensor` represents that the quantization is applied to the entire tensor as a whole. |
| Calibration Algorithm (`algorithm`)| `minmax`, `kl`| 	The calibration algorithm used for the new data type. `minmax` represents the minimum-maximum algorithm, `kl` represents the Kullback-Leibler divergence algorithm. |


To add  4-bit quantization for `Conv2d` in the PyTorch backend. We can modify the `neural_compressor/adaptor/pytorch_cpu.yaml` as follows:

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
The code states that the PyTorch Conv2d Operator has the ability to quantize weights to int4 using the `torch.per_channel_symmetric` quantization scheme, with the supported calibration algorithm being `minmax`. Additionally, the operator can quantize activations to `uint4` using the `torch.per_tensor_symmetric` quantization scheme, with the supported calibration algorithm also being `minmax`.


## Invoke the Operator Kernel According to the Tuning Configuration

One of the tuning configurations generated by the strategy for `Conv2d` looks like as following:

```python

tune_cfg = {
    'op': {
        ('conv', 'Conv2d'): {
            'weight': {
                'dtype': 'int4',
                'algorithm': 'minmax',
                'granularity': 'per_channel',
                'scheme': 'sym'
            },
            'activation': {
                'dtype': 'uint4',
                'quant_mode': 'static',
                'algorithm': 'kl',
                'granularity': 'per_tensor',
                'scheme': 'sym'
            }
        },
```
Now, we can invoke the specified kernel according to the above configurations in the adaptor's `quantize` function. Due to PyTorch currently not having native support for quantization with 4-bit for `Conv2d`, we simulate it numerically by specifying the value ranges of a given data type in the observer. We have implemented it with the following [code](https://github.com/intel/neural-compressor/blob/ad907ab2506514c862f8d79e2109e7407310ceee/neural_compressor/adaptor/pytorch.py#L497-L502):
```diff
    return observer.with_args(qscheme=qscheme,
                              dtype=torch_dtype,
                              reduce_range=(REDUCE_RANGE and scheme == 'asym'),
+                              quant_min=quant_min,
+                              quant_max=quant_max
            )

```

> Note: For PyTorch backend, this simulation only supports N-bit quantization, where N is an integer between 1 and 7.


## Use the New Data Type

Once the new data type has been added to Intel® Neural Compressor, it can be used in the same way as any other data type within the framework. Below is an example of specifying that all `Conv2d` operators should utilize 4-bit quantization:"

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

With this code, all `Conv2d` operators will be quantized to 4-bit, with weight using `int4` and activation using `uint4`.

## Summary
The document outlines the process of adding support for a new data type, such as int4, in Intel® Neural Compressor with minimal changes. It provides instructions and code examples for defining the data type's quantization capabilities, invoking the operator kernel, and using the new data type within the framework. By following the steps outlined in the document, users can extend Intel® Neural Compressor's functionality to accommodate new data types and incorporate them into their quantization workflows.

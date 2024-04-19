Quantization
===============

1. [Quantization Introduction](#quantization-introduction)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [Accuracy Aware Tuning](#with-or-without-accuracy-aware-tuning)
4. [Get Started](#get-started)  
   4.1 [Post Training Quantization](#post-training-quantization)    
   4.2 [Specify Quantization Rules](#specify-quantization-rules)    
   4.3 [Specify Quantization Backend and Device](#specify-quantization-backend-and-device)  
5. [Examples](#examples)

## Quantization Introduction

Quantization is a very popular deep learning model optimization technique invented for improving the speed of inference. It minimizes the number of bits required by converting a set of real-valued numbers into the lower bit data representation, such as int8 and int4, mainly on inference phase with minimal to no loss in accuracy. This way reduces the memory requirement, cache miss rate, and computational cost of using neural networks and finally achieve the goal of higher inference performance. On Intel 3rd Gen Intel® Xeon® Scalable Processors, user could expect up to 4x theoretical performance speedup. We expect further performance improvement with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) on 4th Gen Intel® Xeon® Scalable Processors.

## Quantization Fundamentals

`Affine quantization` and `Scale quantization` are two common range mapping techniques used in tensor conversion between different data types.

The math equation is like: $$X_{int8} = round(Scale \times X_{fp32} + ZeroPoint)$$.

**Affine Quantization**

This is so-called `asymmetric quantization`, in which we map the min/max range in the float tensor to the integer range. Here int8 range is [-128, 127], uint8 range is [0, 255]. 

here:

If INT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 127$ and $ZeroPoint = -128 - X_{f_{min}} / Scale$.

or

If UINT8 is specified, $Scale = (|X_{f_{max}} - X_{f_{min}}|) / 255$ and $ZeroPoint = - X_{f_{min}} / Scale$.

**Scale Quantization**

This is so-called `Symmetric quantization`, in which we use the maximum absolute value in the float tensor as float range and map to the corresponding integer range. 

The math equation is like:

here:

If INT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 127$ and $ZeroPoint = 0$. 

or

If UINT8 is specified, $Scale = max(abs(X_{f_{max}}), abs(X_{f_{min}})) / 255$ and $ZeroPoint = 128$.

*NOTE*

Sometimes the reduce_range feature, that's using 7 bit width (1 sign bit + 6 data bits) to represent int8 range, may be needed on some early Xeon platforms, it's because those platforms may have overflow issues due to fp16 intermediate calculation result when executing int8 dot product operation. After AVX512_VNNI instruction is introduced, this issue gets solved by supporting fp32 intermediate data.

### Quantization Support Matrix

| Framework | Backend Library |  Symmetric Quantization | Asymmetric Quantization |
| :-------------- |:---------------:| ---------------:|---------------:|
| ONNX Runtime | [MLAS](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/mlas) | Weight (int8) | Activation (uint8) |


#### Quantization Scheme
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
+ Asymmetric Quantization
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = min(uint8)  - round(rmin / scale) 

#### Reference
+ MLAS:  [MLAS Quantization](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/onnx_quantizer.py) 

### Quantization Approaches

Quantization has three different approaches:
1) post training dynamic quantization
2) post training static  quantization
3) quantization aware training.

The first two approaches belong to optimization on inference. The last belongs to optimization during training. Currently. ONNX Runtime doesn't support the last one.

#### Post Training Dynamic Quantization

The weights of the neural network get quantized into int8 format from float32 format offline. The activations of the neural network is quantized as well with the min/max range collected during inference runtime.

This approach is widely used in dynamic length neural networks, like NLP model.

#### Post Training Static Quantization

Compared with `post training dynamic quantization`, the min/max range in weights and activations are collected offline on a so-called `calibration` dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

This approach is major quantization approach people should try because it could provide the better performance comparing with `post training dynamic quantization`.

#### Quantization Aware Training

Quantization aware training emulates inference-time quantization in the forward pass of the training process by inserting `fake quant` ops before those quantizable ops. With `quantization aware training`, all weights and activations are `fake quantized` during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while aware of the fact that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

## With or Without Accuracy Aware Tuning

Accuracy aware tuning is one of unique features provided by Intel(R) Neural Compressor, compared with other 3rd party model compression tools. This feature can be used to solve accuracy loss pain points brought by applying low precision quantization and other lossy optimization methods. 

This tuning algorithm creates a tuning space based on user-defined configurations, generates quantized graph, and evaluates the accuracy of this quantized graph. The optimal model will be yielded if the pre-defined accuracy goal is met.

Neural compressor also support to quantize all quantizable ops without accuracy tuning, user can decide whether to tune the model accuracy or not. Please refer to "Get Start" below.

### Working Flow

Currently `accuracy aware tuning` only supports `post training quantization`.

User could refer to below chart to understand the whole tuning flow.

<img src="./imgs/accuracy_aware_tuning_flow.png" width=600 height=480 alt="accuracy aware tuning working flow">


## Get Started

The design philosophy of the quantization interface of Intel(R) Neural Compressor is easy-of-use. It requests user to provide `model`, `calibration dataloader`, and `evaluation function`. Those parameters would be used to quantize and tune the model. 

`model` is the framework model location or the framework model object.

`calibration dataloader` is used to load the data samples for calibration phase. In most cases, it could be the partial samples of the evaluation dataset.

If a user needs to tune the model accuracy, the user should provide `evaluation function`.

`evaluation function` is a function used to evaluate model accuracy. It is a optional. This function should be same with how user makes evaluation on fp32 model, just taking `model` as input and returning a scalar value represented the evaluation accuracy.

User could execute:
### Post Training Quantization

1. Without Accuracy Aware Tuning

This means user could leverage Intel(R) Neural Compressor to directly generate a fully quantized model without accuracy aware tuning. It's user responsibility to ensure the accuracy of the quantized model meets expectation. Intel(R) Neural Compressor supports `Post Training Static Quantization` and `Post Training Dynamic Quantization`.

``` python
from neural_compressor_ort.quantization import StaticQuantConfig, DynamicQuantConfig, quantize
from neural_compressor_ort.quantization.calibrate import CalibrationDataReader

class DataReader(CalibrationDataReader):
    def get_next(self):
        ...

    def rewind(self):
        ...

calibration_data_reader = DataReader() # only needed by StaticQuantConfig
config = StaticQuantConfig(calibration_data_reader) # or config = DynamicQuantConfig()
quantize(model, q_model_path, config)
```

2. With Accuracy Aware Tuning

This means user could leverage the advance feature of Intel(R) Neural Compressor to tune out a best quantized model which has best accuracy and good performance. User should provide `eval_func`.

``` python
from neural_compressor_ort.common.base_tuning import Evaluator, TuningConfig
from neural_compressor_ort.quantization import (
    CalibrationDataReader,
    GPTQConfig,
    RTNConfig,
    autotune,
    get_woq_tuning_config,
)

class DataReader(CalibrationDataReader):
    def get_next(self):
        ...

    def rewind(self):
        ...

data_reader = DataReader()

# TuningConfig can accept:
# 1) a set of candidate configs like TuningConfig(config_set=[RTNConfig(weight_bits=4), GPTQConfig(weight_bits=4)])
# 2) one config with a set of candidate parameters like TuningConfig(config_set=[GPTQConfig(weight_group_size=[32, 64])])
# 3) our pre-defined config set like TuningConfig(config_set=get_woq_tuning_config())
custom_tune_config = TuningConfig(config_set=[RTNConfig(weight_bits=4), GPTQConfig(weight_bits=4)])
best_model = autotune(
    model_input=model,
    tune_config=custom_tune_config,
    eval_fn=eval_fn,
    calibration_data_reader=data_reader,
)
```

### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator name. Users can use `set_local` API of configs to achieve the above purpose by below code:

```python
fp32_config = GPTQConfig(weight_dtype="fp32")
quant_config = GPTQConfig(
    weight_bits=4,
    weight_dtype="int",
    weight_sym=False,
    weight_group_size=32,
)
quant_config.set_local("/h.4/mlp/fc_out/MatMul", fp32_config)
```


### Specify Quantization Backend and Device

Neural-Compressor will detect the hardware and software status automatically to decide which backend should be used. The priority is: accelerator > GPU > CPU.


<table class="center">
    <thead>
        <tr>
            <th>Framework</th>
            <th>Backend</th>
            <th>Backend Library</th>
            <th>Support Device(cpu as default)</th> 
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5" align="left">ONNX Runtime</td>
            <td align="left">CPUExecutionProvider</td>
            <td align="left">MLAS</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">TensorrtExecutionProvider</td>
            <td align="left">TensorRT</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">CUDAExecutionProvider</td>
            <td align="left">CUDA</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">DnnlExecutionProvider</td>
            <td align="left">OneDNN</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">DmlExecutionProvider*</td>
            <td align="left">OneDNN</td>
            <td align="left">npu</td>
        </tr>
    </tbody>
</table>
<br>
<br>

> ***Note***
> 
> DmlExecutionProvider support works as experimental, please expect exceptions.
> 
> Known limitation: the batch size of onnx models has to be fixed to 1 for DmlExecutionProvider, no multi-batch and dynamic batch support yet.


## Examples

User could refer to [examples](../../examples/onnxrt) on how to quantize a new model.
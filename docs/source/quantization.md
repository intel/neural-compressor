Quantization
===============

1. [Quantization Introduction](#quantization-introduction)
2. [Quantization Fundamentals](#quantization-fundamentals)
3. [Accuracy Aware Tuning](#accuracy-aware-tuning)
4. [Supported Feature Matrix](#supported-feature-matrix)
5. [Get Started](#get-started)  
   5.1 [Post Training Quantization](#post-training-quantization)   
   5.2 [Quantization Aware Training](#quantization-aware-training-1)  
   5.3 [Specify Quantization Rules](#specify-quantization-rules)  
   5.4 [Specify Quantization Recipes](#specify-quantization-recipes)  
   5.5 [Specify Quantization Backend and Device](#specify-quantization-backend-and-device)  
6. [Examples](#examples)

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
| TensorFlow    | [oneDNN](https://github.com/oneapi-src/oneDNN) | Activation (int8/uint8), Weight (int8) | - |
| PyTorch         | [FBGEMM](https://github.com/pytorch/FBGEMM) | Activation (uint8), Weight (int8) | Activation (uint8) |
| PyTorch(IPEX) | [oneDNN](https://github.com/oneapi-src/oneDNN)  | Activation (int8/uint8), Weight (int8) | - |
| MXNet           | [oneDNN](https://github.com/oneapi-src/oneDNN)  | Activation (int8/uint8), Weight (int8) | - |
| ONNX Runtime | [MLAS](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/mlas) | Weight (int8) | Activation (uint8) |

#### Quantization Scheme in TensorFlow
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))

#### Quantization Scheme in PyTorch
+ Symmetric Quantization
    + int8: scale = max(abs(rmin), abs(rmax)) / (float(max(int8) - min(int8)) / 2)
    + uint8: scale = max(abs(rmin), abs(rmax)) / (float(max(int8) - min(int8)) / 2)
+ Asymmetric Quantization
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = min(uint8)  - round(rmin / scale)

#### Quantization Scheme in IPEX
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))

#### Quantization Scheme in MXNet
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
    + uint8: scale = max(rmin, rmax) / (max(uint8) - min(uint8))

#### Quantization Scheme in ONNX Runtime
+ Symmetric Quantization
    + int8: scale = 2 * max(abs(rmin), abs(rmax)) / (max(int8) - min(int8) - 1)
+ Asymmetric Quantization
    + uint8: scale = (rmax - rmin) / (max(uint8) - min(uint8)); zero_point = min(uint8)  - round(rmin / scale) 

#### Reference
+ oneDNN: [Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/content/www/us/en/develop/articles/lower-numerical-precision-deep-learning-inference-and-training.html)
+ FBGEMM: [FBGEMM Quantization](https://github.com/pytorch/pytorch/blob/master/torch/quantization/observer.py)
+ MLAS:  [MLAS Quantization](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/onnx_quantizer.py) 

### Quantization Approaches

Quantization has three different approaches:
1) post training dynamic quantization
2) post training static  quantization
3) quantization aware training.

The first two approaches belong to optimization on inference. The last belongs to optimization during training.

#### Post Training Dynamic Quantization

The weights of the neural network get quantized into int8 format from float32 format offline. The activations of the neural network is quantized as well with the min/max range collected during inference runtime.

This approach is widely used in dynamic length neural networks, like NLP model.

#### Post Training Static Quantization

Compared with `post training dynamic quantization`, the min/max range in weights and activations are collected offline on a so-called `calibration` dataset. This dataset should be able to represent the data distribution of those unseen inference dataset. The `calibration` process runs on the original fp32 model and dumps out all the tensor distributions for `Scale` and `ZeroPoint` calculations. Usually preparing 100 samples are enough for calibration.

This approach is major quantization approach people should try because it could provide the better performance comparing with `post training dynamic quantization`.

#### Quantization Aware Training

Quantization aware training emulates inference-time quantization in the forward pass of the training process by inserting `fake quant` ops before those quantizable ops. With `quantization aware training`, all weights and activations are `fake quantized` during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while aware of the fact that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

## Accuracy Aware Tuning

Accuracy aware tuning is one of unique features provided by Intel(R) Neural Compressor, compared with other 3rd party model compression tools. This feature can be used to solve accuracy loss pain points brought by applying low precision quantization and other lossy optimization methods. 

This tuning algorithm creates a tuning space by querying framework quantization capability and model structure, selects the ops to be quantized by the tuning strategy, generates quantized graph, and evaluates the accuracy of this quantized graph. The optimal model will be yielded if the pre-defined accuracy goal is met.

Neural compressor also support to quantize all quantizable ops without accuracy tuning, user can decide whether to tune the model accuracy or not. Please refer to "Get Start" below.

### Working Flow

Currently `accuracy aware tuning` supports `post training quantization`, `quantization aware training`.

User could refer to below chart to understand the whole tuning flow.

<img src="./imgs/accuracy_aware_tuning_flow.png" width=600 height=480 alt="accuracy aware tuning working flow">

## Supported Feature Matrix

Quantization methods include the following three types:
<table class="center">
    <thead>
        <tr>
            <th>Types</th>
            <th>Quantization</th>
            <th>Dataset Requirements</th>
            <th>Framework</th>
            <th>Backend</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3" align="center">Post-Training Static Quantization (PTQ)</td>
            <td rowspan="3" align="center">weights and activations</td>
            <td rowspan="3" align="center">calibration</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch Eager</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch FX</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">TensorFlow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
        <tr>
            <td align="center">ONNX Runtime</td>
            <td align="center"><a href="https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py">QLinearops/QDQ</a></td>
        </tr>
        <tr>
            <td rowspan="2" align="center">Post-Training Dynamic Quantization</td>
            <td rowspan="2" align="center">weights</td>
            <td rowspan="2" align="center">none</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch eager mode</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch fx mode</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">ONNX Runtime</td>
            <td align="center"><a href="https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py">QIntegerops</a></td>
        </tr>  
        <tr>
            <td rowspan="2" align="center">Quantization-aware Training (QAT)</td>
            <td rowspan="2" align="center">weights and activations</td>
            <td rowspan="2" align="center">fine-tuning</td>
            <td align="center">PyTorch</td>
            <td align="center"><a href="https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization">PyTorch eager mode</a>/<a href="https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization">PyTorch fx mode</a>/<a href="https://github.com/intel/intel-extension-for-pytorch">IPEX</a></td>
        </tr>
        <tr>
            <td align="center">TensorFlow</td>
            <td align="center"><a href="https://github.com/tensorflow/tensorflow">TensorFlow</a>/<a href="https://github.com/Intel-tensorflow/tensorflow">Intel TensorFlow</a></td>
        </tr>
    </tbody>
</table>
<br>
<br>

## Get Started

The design philosophy of the quantization interface of Intel(R) Neural Compressor is easy-of-use. It requests user to provide `model`, `calibration dataloader`, and `evaluation function`. Those parameters would be used to quantize and tune the model. 

`model` is the framework model location or the framework model object.

`calibration dataloader` is used to load the data samples for calibration phase. In most cases, it could be the partial samples of the evaluation dataset.

If a user needs to tune the model accuracy, the user should provide either `evaluation function` or `evaluation dataloader` `evaluation metric`. If the user won't to tune the model accuracy, then the user should provide neither `evaluation function` nor `evaluation dataloader` `evaluation metric`.

`evaluation function` is a function used to evaluate model accuracy. It is a optional. This function should be same with how user makes evaluation on fp32 model, just taking `model` as input and returning a scalar value represented the evaluation accuracy.

`evaluation dataloader` is a data loader for evaluation. It is iterable and should yield a tuple of (input, label). The input could be a object, list, tuple or dict, depending on user implementation, as well as it can be taken as model input. The label should be able to take as input of supported metrics. If this parameter is not None, user needs to specify pre-defined evaluation metrics through configuration file and should set "eval_func" parameter as None. Tuner will combine model, eval_dataloader and pre-defined metrics to run evaluation process.

`evaluation metric` is an object to compute the metric to evaluating the performance of the model or a dict of built-in metric configures, neural_compressor will initialize this class when evaluation. `evaluation metric` must be supported by neural compressor. Please refer to [metric.md](metric.md).

User could execute:
### Post Training Quantization

1. Without Accuracy Aware Tuning

This means user could leverage Intel(R) Neural Compressor to directly generate a fully quantized model without accuracy aware tuning. It's user responsibility to ensure the accuracy of the quantized model meets expectation. Intel(R) Neural Compressor support `Post Training Static Quantization` and `Post Training Dynamic Quantization`.

``` python
# main.py

# Original code
model = ResNet50()
val_dataset = ...
val_dataloader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    ping_memory=True,
)

# Quantization code
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

conf = (
    PostTrainingQuantConfig()
)  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")
q_model = quantization.fit(
    model=model,
    conf=conf,
    calib_dataloader=val_dataloader,
)
q_model.save("./output")
```

2. With Accuracy Aware Tuning

This means user could leverage the advance feature of Intel(R) Neural Compressor to tune out a best quantized model which has best accuracy and good performance. User should provide either `eval_func` or `eval_dataloader` `eval_metric`.

``` python
# main.py


# Original code
def validate(val_loader, model, criterion, args):
    ...
    return top1.avg


model = ResNet50()
val_dataset = ...
val_dataloader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    ping_memory=True,
)

# Quantization code
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

conf = PostTrainingQuantConfig()
q_model = quantization.fit(
    model=model,
    conf=conf,
    calib_dataloader=val_dataloader,
    eval_func=validate,
)
q_model.save("./output")
```
or

```python
from neural_compressor.metric import METRICS

metrics = METRICS("pytorch")
top1 = metrics["topk"]()
q_model = quantization.fit(
    model=model,
    conf=conf,
    calib_dataloader=val_dataloader,
    eval_dataloader=val_dataloader,
    eval_metric=top1,
)
```
### Quantization Aware Training

1. Without Accuracy Aware Tuning
This method only requires the user to call the callback function during the training process. After the training is completed, after the training is completed, Neural Compressor will convert to quantized model. 

```python
# main.py

# Original code
model = ResNet50()
train_dataset = ...
train_dataloader = torch.utils.data.Dataloader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    ping_memory=True,
)
criterion = ...


# Quantization code
def train_func(model): ...


from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression

conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model, conf)
compression_manager.callbacks.on_train_begin()
model = compression_manager.model
train_func(model)
compression_manager.callbacks.on_train_end()
compression_manager.save("./output")
```

2. With Accuracy Aware Tuning
This method requires the user to provide training function and evaluation function to Neural Compressor, and in training function, the user should call the callback function.

```python
# main.py

# Original code
model = ResNet50()
val_dataset = ...
val_dataloader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    ping_memory=True,
)
criterion = ...


def validate(val_loader, model, criterion, args):
    ...
    return top1.avg


# Quantization code
def train_func(model):
    ...
    return model  # user should return a best performance model here


from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression, fit

conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model, conf)
q_model = fit(compression_manager=compression_manager, train_func=train_func, eval_func=validate)
compression_manager.save("./output")
```

### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator name or operator type. Users can set `op_name_dict` and `op_type_dict` in config class to achieve the above purpose.

1. Example of `op_name_dict`
```python
op_name_dict = {
    "layer1.0.conv1": {
        "activation": {
            "dtype": ["fp32"],
        },
        "weight": {
            "dtype": ["fp32"],
        },
    },
    "layer2.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"],
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"],
        },
    },
}
conf = PostTrainingQuantConfig(op_name_dict=op_name_dict)
```
2. Example of `op_type_dict`
```python
op_type_dict = {"Conv": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
conf = PostTrainingQuantConfig(op_type_dict=op_type_dict)
```

### Specify Quantization Recipes
Intel(R) Neural Compressor support some quantization recipes. Users can set `recipes` in config class to achieve the above purpose. (`fast_bias_correction` and `weight_correction` is working in progress.)

| Recipes | PyTorch |  Tensorflow | ONNX Runtime |
| :---------------- |:---------------:| ---------------:|---------------:|
| smooth_quant      | ✅ | N/A | ✅ |
| smooth_quant_args | ✅ | N/A | ✅ |
| fast_bias_correction | N/A | N/A | N/A |
| weight_correction | N/A | N/A | N/A |
| first_conv_or_matmul_quantization | N/A | ✅ | ✅ |
| last_conv_or_matmul_quantization | N/A | ✅ | ✅ |
| pre_post_process_quantization | N/A | N/A | ✅ |
| gemm_to_matmul | N/A | N/A | ✅ |
| graph_optimization_level | N/A | N/A | ✅ |
| add_qdq_pair_to_weight | N/A | N/A | ✅ |
| optypes_to_exclude_output_quant | N/A | N/A | ✅ |
| dedicated_qdq_pair | N/A | N/A | ✅ |

Example of recipe:
```python
recipes = {
    "smooth_quant": True,
    "smooth_quant_args": {
        "alpha": 0.5,
    },  # default value is 0.5
    "fast_bias_correction": False,
}
conf = PostTrainingQuantConfig(recipes=recipes)
```

### Specify Quantization Backend and Device
Intel(R) Neural Compressor support multi-framework: PyTorch, Tensorflow, ONNX Runtime and MXNet. The neural compressor will automatically determine which framework to use based on the model type, but for backend and device, users need to set it themselves in configure object.

<table class="center">
    <thead>
        <tr>
            <th>Framework</th>
            <th>Backend</th>
            <th>Backend Library</th>
            <th>Backend Value</th>
            <th>Support Device(cpu as default)</th> 
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" align="left">PyTorch</td>
            <td align="left">FX</td>
            <td align="left">FBGEMM</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">IPEX</td>
            <td align="left">OneDNN</td>
            <td align="left">"ipex"</td>
            <td align="left">cpu | xpu</td>
        </tr>
        <tr>
            <td rowspan="5" align="left">ONNX Runtime</td>
            <td align="left">CPUExecutionProvider</td>
            <td align="left">MLAS</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">TensorrtExecutionProvider</td>
            <td align="left">TensorRT</td>
            <td align="left">"onnxrt_trt_ep"</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">CUDAExecutionProvider</td>
            <td align="left">CUDA</td>
            <td align="left">"onnxrt_cuda_ep"</td>
            <td align="left">gpu</td>
        </tr>
        <tr>
            <td align="left">DnnlExecutionProvider</td>
            <td align="left">OneDNN</td>
            <td align="left">"onnxrt_dnnl_ep"</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">DmlExecutionProvider*</td>
            <td align="left">OneDNN</td>
            <td align="left">"onnxrt_dml_ep"</td>
            <td align="left">npu</td>
        </tr>
        <tr>
            <td rowspan="2" align="left">Tensorflow</td>
            <td align="left">Tensorflow</td>
            <td align="left">OneDNN</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
        </tr>
        <tr>
            <td align="left">ITEX</td>
            <td align="left">OneDNN</td>
            <td align="left">"itex"</td>
            <td align="left">cpu | gpu</td>
        </tr>  
        <tr>
            <td align="left">MXNet</td>
            <td align="left">OneDNN</td>
            <td align="left">OneDNN</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
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

Examples of configure:
```python
# run with PT FX on CPU
conf = PostTrainingQuantConfig()
```
```python
# run with IPEX on CPU
conf = PostTrainingQuantConfig(backend="ipex")
# run with IPEX on XPU
conf = PostTrainingQuantConfig(backend="ipex", device="xpu")
```
```python
# run with ONNXRT CUDAExecutionProvider on GPU
conf = PostTrainingQuantConfig(backend="onnxrt_cuda_ep", device="gpu")
```
```python
# run with ONNXRT DmlExecutionProvider on NPU
conf = PostTrainingQuantConfig(backend="onnxrt_dml_ep", device="npu")
```
```python
# run with ITEX on GPU
conf = PostTrainingQuantConfig(backend="itex", device="gpu")
```

## Examples

User could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/README.md) on how to quantize a new model.
If user wants to quantize an onnx model with npu, please refer to this [example](../../examples/onnxrt/image_recognition/onnx_model_zoo/shufflenet/quantization/ptq_static/README.md). If user wants to quantize a pytorch model with xpu, please refer to this [example](../../examples/pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex/README.md).

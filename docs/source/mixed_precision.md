Mixed Precision
===============

1. [Introduction](#introduction)
2. [Mixed Precision Support Matrix](#mixed-precision-support-matrix)
3. [Get Started with Mixed Precision API](#get-start-with-mixed-precision-api)
4. [Examples](#examples)

## Introduction

The recent growth of Deep Learning has driven the development of more complex models that require significantly more compute and memory capabilities. Several low precision numeric formats have been proposed to address the problem. Google's [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and the [FP16: IEEE](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) half-precision format are two of the most widely used sixteen bit formats. [Mixed precision](https://arxiv.org/abs/1710.03740) training and inference using low precision formats have been developed to reduce compute and bandwidth requirements.

The recently launched 3rd Gen Intel® Xeon® Scalable processor (codenamed Cooper Lake), featuring Intel® Deep Learning Boost, is the first general-purpose x86 CPU to support the bfloat16 format. Specifically, three new bfloat16 instructions are added as a part of the AVX512_BF16 extension within Intel Deep Learning Boost: VCVTNE2PS2BF16, VCVTNEPS2BF16, and VDPBF16PS. The first two instructions allow converting to and from bfloat16 data type, while the last one performs a dot product of bfloat16 pairs. Further details can be found in the [hardware numerics document](https://software.intel.com/content/www/us/en/develop/download/bfloat16-hardware-numerics-definition.html) published by Intel.

<p align="center" width="100%">
    <img src="./imgs/data_format.png" alt="Architecture" height=230>
</p>

## Mixed Precision Support Matrix
<table class="center">
    <thead>
        <tr>
            <th>Framework</th>
            <th>Backend</th>
            <th>Backend Library</th>
            <th>Backend Value</th>
            <th>Support Device(cpu as default)</th> 
            <th>Support BF16</th>
            <th>Support FP16</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2" align="left">PyTorch</td>
            <td align="left">FX</td>
            <td align="left">FBGEMM</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
            <td align="left">&#10004;</td>
            <td align="left">:x:</td>
        </tr>
        <tr>
            <td align="left">IPEX</td>
            <td align="left">OneDNN</td>
            <td align="left">"ipex"</td>
            <td align="left">cpu</td>
            <td align="left">&#10004;</td>
            <td align="left">:x:</td>
        </tr>
        <tr>
            <td rowspan="3" align="left">ONNX Runtime</td>
            <td align="left">CPUExecutionProvider</td>
            <td align="left">MLAS</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
            <td align="left">:x:</td>
            <td align="left">:x:</td>
        </tr>
        <tr>
            <td align="left">TensorrtExecutionProvider</td>
            <td align="left">TensorRT</td>
            <td align="left">"onnxrt_trt_ep"</td>
            <td align="left">gpu</td>
            <td align="left">:x:</td>
            <td align="left">:x:</td>
        </tr>
        <tr>
            <td align="left">CUDAExecutionProvider</td>
            <td align="left">CUDA</td>
            <td align="left">"onnxrt_cuda_ep"</td>
            <td align="left">gpu</td>
            <td align="left">&#10004;</td>
            <td align="left">&#10004;</td>
        </tr>
        <tr>
            <td rowspan="2" align="left">Tensorflow</td>
            <td align="left">Tensorflow</td>
            <td align="left">OneDNN</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
            <td align="left">&#10004;</td>
            <td align="left">:x:</td>
        </tr>
        <tr>
            <td align="left">ITEX</td>
            <td align="left">OneDNN</td>
            <td align="left">"itex"</td>
            <td align="left">cpu | gpu</td>
            <td align="left">&#10004;</td>
            <td align="left">:x:</td>
        </tr>  
        <tr>
            <td align="left">MXNet</td>
            <td align="left">OneDNN</td>
            <td align="left">OneDNN</td>
            <td align="left">"default"</td>
            <td align="left">cpu</td>
            <td align="left">&#10004;</td>
            <td align="left">:x:</td>
        </tr>
    </tbody>
</table>


### Hardware and Software requests for **BF16**
- TensorFlow
  1. Hardware: CPU supports `avx512_bf16` instruction set.
  2. Software: intel-tensorflow >= [2.3.0](https://pypi.org/project/intel-tensorflow/2.3.0/).
- PyTorch
  1. Hardware: CPU supports `avx512_bf16` instruction set.
  2. Software: torch >= [1.11.0](https://download.pytorch.org/whl/torch_stable.html).
- ONNX Runtime
  1. Hardware: GPU, set 'device' of config to 'gpu' and 'backend' to 'onnxrt_cuda_ep'.
  2. Software: onnxruntime-gpu.

### Hardware and Software requests for **FP16**
- ONNX Runtime
  1. Hardware: GPU, set 'device' of config to 'gpu' and 'backend' to 'onnxrt_cuda_ep'.
  2. Software: onnxruntime-gpu.

### During quantization mixed precision
During quantization, if the hardware support BF16, the conversion is default enabled. So you may get an INT8/BF16/FP32 mixed precision model on those hardware. FP16 can be executed if 'device' of config is 'gpu'.
Please refer to this [document](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_mixed_precision.md) for its workflow.

### Accuracy-driven mixed precision
BF16/FP16 conversion may lead to accuracy drop. Intel® Neural Compressor provides an accuracy-driven tuning function to reduce accuracy loss, 
which will fallback converted ops to FP32 automatically to get better accuracy. To enable this function, users only to provide 
`evaluation function` or (`evaluation dataloader` plus `evaluation metric`) for [mixed precision inputs](https://github.com/intel/neural-compressor/blob/master/neural_compressor/mix_precision.py).   
To be noticed, IPEX backend doesn't support accuracy-driven mixed precision.  

## Get Started with Mixed Precision API

To get a bf16/fp16 model, users can use the Mixed Precision API as follows.

- BF16:

```python
from neural_compressor import mix_precision
from neural_compressor.config import MixedPrecisionConfig

conf = MixedPrecisionConfig() # default precision is bf16
converted_model = mix_precision.fit(model, conf=conf)
converted_model.save('./path/to/save/')
```

- FP16:

```python
from neural_compressor import mix_precision
from neural_compressor.config import MixedPrecisionConfig

conf = MixedPrecisionConfig(
        backend='onnxrt_cuda_ep',
        device='gpu',
        precisions='fp16')
converted_model = mix_precision.fit(model, conf=conf)
converted_model.save('./path/to/save/')
```
  
## Examples

- Quick started with [helloworld example](/examples/helloworld/tf_example3)
- PyTorch [ResNet18](/examples/pytorch/image_recognition/torchvision_models/mixed_precision/resnet18)
- IPEX [DistilBERT base](/examples/pytorch/nlp/huggingface_models/question-answering/mixed_precision/ipex)
- Tensorflow [ResNet50](/examples/tensorflow/image_recognition/tensorflow_models/resnet50_v1/mixed_precision) 
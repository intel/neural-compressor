Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch tuning results with Intel® Neural Compressor.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/pytorch-imagenet
train  val
ls /path/to/onnx-imagenet-validation
ILSVRC2012_img_val val.txt
```

# Run
### 1. To get the exported model: 

Run run_export.sh to get ONNX model from PyTorch model.
```bash
# export fp32 model
bash run_export.sh --input_model=resnet50 --dtype=fp32 --dataset_location=/path/to/pytorch-imagenet --output_model=resnet50-fp32.onnx
# export int8 model
bash run_export.sh --input_model=resnet50 --dtype=int8 --quant_format=[QDQ|QOperator] --dataset_location=/path/to/pytorch-imagenet --output_model=resnet50-int8.onnx --approach=[static|dynamic]
```

### 2. To get the benchmark of exported and tuned models, includes Batch_size and Throughput: 
Run run_benchmark.sh to benchmark the accuracy and performance of ONNX models and PyTorch model.
```bash
# benchmark ONNX model
bash run_benchmark.sh --input_model=[resnet50-fp32.onnx|resnet50-int8.onnx] --dataset_location=/path/to/onnx-imagenet-validation --mode=[accuracy|performance] --batch_size=[16]
# benchmark PyTorch model
bash run_benchmark.sh --input_model=[resnet50|/path/to/saved_results] --dataset_location=/path/to/pytorch-imagenet --mode=[accuracy|performance] --int8=[true|false] --batch_size=[16]
```

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

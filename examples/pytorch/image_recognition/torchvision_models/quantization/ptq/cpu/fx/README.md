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
ls /path/to/imagenet
train  val
```

# Run

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. ResNet50

```shell
python main.py -t -a resnet50 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnet50 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnet50 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 2. ResNet18

```shell
python main.py -t -a resnet18 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnet18 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnet18 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 3. ResNeXt101_32x8d

```shell
python main.py -t -a resnext101_32x8d --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnext101_32x8d --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnext101_32x8d --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 4. InceptionV3

```shell
python main.py -t -a inception_v3 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=inception_v3 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=inception_v3 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 5. Mobilenet_v2

```shell
python main.py -t -a mobilenet_v2 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=mobilenet_v2 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=mobilenet_v2 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 6. Efficientnet_b0

```shell
python main.py -t -a efficientnet_b0 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=efficientnet_b0 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=efficientnet_b0 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

> **Note**
>
> To reduce tuning time and get the result faster, the `efficientnet_b0` model uses 
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) by default.


### 7. Efficientnet_b3

```shell
python main.py -t -a efficientnet_b3 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=efficientnet_b3 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=efficientnet_b3 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

> **Note**
>
> To reduce tuning time and get the result faster, the `efficientnet_b3` model uses 
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) by default.
### 8. Efficientnet_b7

```shell
python main.py -t -a efficientnet_b7 --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=efficientnet_b7 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=efficientnet_b7 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

> **Note**
>
> To reduce tuning time and get the result faster, the `efficientnet_b7` model uses 
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) by default.


# Saving and Loading Model

* Saving model:
  After tuning with Neural Compressor, we can get neural_compressor.model:

```
from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                            conf,
                            calib_dataloader=val_loader,
                            eval_func=eval_func)
```

Here, `q_model` is the Neural Compressor model class, so it has "save" API:

```python
q_model.save("Path_to_save_quantized_model")
```

* Loading model:

```python
from neural_compressor.utils.pytorch import load
quantized_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
                        model,
                        dataloader=val_loader)
```
Here, `dataloader` is used to get example_inputs for `torch.fx` to trace the model. You can also pass in `example_inputs` instead. For torch version < 1.13.0, you can ignore this parameter.

Please refer to [Sample code](./main.py).

Tutorial
=========================================

This tutorial will introduce step by step instructions on how to integrate models with Intel® Low Precision Optimization Tool.

Intel® Low Precision Optimization Tool supports three usages:

1. Fully yaml configuration: User specifies all the info through yaml, including dataloaders used in calibration and evaluation
   phases and quantization tuning settings.

   For this usage, only model parameter is mandotory.

2. Partial yaml configuration: User specifies dataloaders used in calibration and evaluation phase by code.
   The tool provides built-in dataloaders and evaluators, user just need provide a dataset implemented __iter__ or
   __getitem__ methods and invoke dataloader() with dataset as input parameter before calling tune().

   After that, User specifies fp32 "model", calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader".
   The calibrated and quantized model is evaluated with "eval_dataloader" with evaluation metrics specified
   in the configuration file. The evaluation tells the tuner whether the quantized model meets
   the accuracy criteria. If not, the tuner starts a new calibration and tuning flow.

   For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.

3. Partial yaml configuration: User specifies dataloaders used in calibration phase by code.
   This usage is quite similar with b), just user specifies a custom "eval_func" which encapsulates
   the evaluation dataset by itself.
   The calibrated and quantized model is evaluated with "eval_func". The "eval_func" tells the
   tuner whether the quantized model meets the accuracy criteria. If not, the Tuner starts a new
   calibration and tuning flow.

   For this usage, model, q_dataloader and eval_func parameters are mandotory

# General Steps

### 1. Usage Choose

User need choose corresponding usage according to code. For example, if user wants to minmal code changes, then the first usage
is recommended. If user wants to leverage existing evaluation function, then the third usage is recommended. If user has no existing
evaluation function and the metric used is supported by ilit, then the second usage is recommended.

### 2. Write yaml config file

Copy [template.yaml](../examples/template.yaml) to work directory and modify correspondingly.

Below is an example for beginner.

```
framework:
  - name: pytorch

tuning:
    metric
      - topk: 1
    accuracy_criterion:
      - relative: 0.01
    timeout: 0
    random_seed: 9527
```

Below is an example for advance user, which constrain the tuning space by specifing calibration, quantization, tuning.ops fields accordingly.

```
framework:
  - name: tensorflow
    inputs: input
    outputs: MobilenetV1/Predictions/Reshape_1

calibration:
  - iterations: 10, 50
    algorithm:
      - weight:  minmax
        activation: minmax

quantization:
  - weight:
      - granularity: per_channel
        scheme: asym
        dtype: int8
    activation:
      - granularity: per_tensor
        scheme: asym
        dtype: int8

tuning:
    metric:
      - topk: 1
    accuracy_criterion:
      - relative:  0.01
    objective: performance
    timeout: 36000
    ops: {
           'conv1': {
             'activation':  {'dtype': ['uint8', 'fp32'], 'algorithm': ['minmax', 'kl'], 'scheme':['sym']},
             'weight': {'dtype': ['int8', 'fp32'], 'algorithm': ['kl']}
           }
         }

```

### 3. Integration with Intel® Low Precision Optimization Tool

   a. Check if calibration or evaluation dataloader in user code meets Intel® Low Precision Optimization Tool requirements, that is whether it returns a tuple of (input, label). In classification networks, its dataloader usually yield output like this. As calication dataset does not need to have label, user need wrapper the loader to return a tuple of (input, _) for Intel® Low Precision Optimization Tool on this case. In object detection or NLP or recommendation networks, its dataloader usually yield output not like this, user need wrapper the loder to return a tuple of (input, label), in which "input" may be a object, a tuple or a dict.

   b. Check if model in user code could be directly feed "input" got from #a. If not, user need wrapper the model to take "input" as input.

   c. If user choose the first use case, that is using Intel® Low Precision Optimization Tool build-in metrics. User need ensure metric built in Intel® Low Precision Optimization Tool could take output of model and label of eval_dataloader as input.


# Features
| Features | Link |
| ------ | ------ |
| Unified dataloader and metric |  [dataloader_metric.md](./dataloader_metric.md)|
| QAT for PyTorch (Experimental) | [qat_calibration_mode.md](./qat_calibration_mode.md)| 
| BF16 of TensorFlow | [bf16_convert.md](./bf16_convert.md)| 

 # Examples
| Examples Tutorials |
| ------ | 
|[Hello World examples for quick start](../examples/helloworld/README.md)| 
|[PyTorch imagenet recognition/imagenet](../examples/pytorch/image_recognition/imagenet/cpu/PTQ/README.md)| 
|[PyTorch imagenet recognition/peleenet](../examples/pytorch/image_recognition/peleenet/README.md)|
|[PyTorch imagenet recognition/resnest50](../examples/pytorch/image_recognition/resnest/README.md)|
|[PyTorch imagenet recognition/se_resnext50](../examples/pytorch/image_recognition/se_resnext/README.md)|
|[PyTorch language translation](../examples/pytorch/language_translation/README.md)| 
|[PyTorch object detection](../examples/pytorch/object_detection/yolo_v3/README.md)|
|[PyTorch recommendation](../examples/pytorch/recommendation/README.md)| 
|[TensorFlow Image Recognition](../examples/tensorflow/image_recognition/README.md)|
|[TensorFlow object detection](../examples/tensorflow/object_detection/README.md)|
|[TensorFlow recommendation](../examples/tensorflow/recommendation/wide_deep_large_ds/WND_README.md)|
|[TensorFlow style_transfer](../examples/tensorflow/style_transfer/README.md)|
|[MxNet Image Recognition](../examples/mxnet/image_recognition/README.md)|
|[Mxnet language translation](../examples/mxnet/language_translation/README.md)|
|[MxNet object detection](../examples/mxnet/object_detection/README.md)|


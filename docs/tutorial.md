Tutorial
=========================================

This tutorial will introduce step by step instructions on how to integrate models with Intel® Low Precision Optimization Tool.

Intel® Low Precision Optimization Tool supports two usages:

1. User specifies fp32 "model", calibration dataset "q_dataloader", evaluation dataset "eval_dataloader" and metric in tuning.metric field of model-specific yaml config file.

2. User specifies fp32 "model", calibration dataset "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset and metric by itself.

# General Steps

### 1. Usage Choose

If metric used by user model is supported by Intel® Low Precision Optimization Tool, user could choose the first usage.

If metric used by user model is NOT supported by Intel® Low Precision Optimization Tool, user need choose the second usage.

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


# Detail Examples

### MxNet

* [NLP](../examples/mxnet/language_translation/README.md), including Bert MRPC and Bert Squad tasks.

* [Image Recognition](../examples/mxnet/image_recognition/README.md), including ResNet50 V1, ResNet18, MobileNet V1, ResNet18, SqueezeNet V1 examples.

* [Object Detection](../examples/mxnet/object_detection/README.md), including SSD-ResNet50, SSD-MobileNet V1 examples.

### PyTorch

* [Recommendation](../examples/pytorch/recommendation/README.md), including DLRM.

* [NLP](../examples/pytorch/language_translation/README.md) Including all 10 BERT task exampls.

* [Image Recognition](../examples/pytorch/image_recognition/resnet/README.md) Including ResNet18, ResNet50 and ResNet101 examples.

* [Image Recognition QAT](../examples/pytorch/image_recognition/resnet_qat/README.md) Including ResNet18, ResNet50 and ResNet101 examples.

### TensorFlow

* [Image Recognition](../examples/tensorflow/image_recognition/README.md), including ResNet50 V1, ResNet50 V1.5, ResNet101, MobileNet V1, MobileNet V2, Inception V1, Inception V2, Inception V3, Inception V4, Inception ResNet V2 examples.

* [Object Detection](../examples/tensorflow/object_detection/README.md), including SSD ResNet50 example.

Tutorial
=========================================

This tutorial will introduce step by step instructions on how to integrate models with Intel® Low Precision Optimization Tool.

Intel® Low Precision Optimization Tool supports three usages:

1. Fully yaml configuration: User specifies all the info through yaml, including dataloaders used in calibration and evaluation
   phases and quantization tuning settings.

   For this usage, only model parameter is mandotory.

2. Partial yaml configuration: User specifies dataloaders used in calibration and evaluation phase by code.
   The tool provides built-in dataloaders and evaluators, user just need provide a dataset implemented __iter__ or
   __getitem__ methods and invoke dataloader() with dataset as input parameter to create lpot dataloader before calling quantizer().

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
evaluation function and the metric used is supported by lpot, then the second usage is recommended.

### 2. Write yaml config file

Copy [ptq.yaml](../lpot/template/ptq.yaml) or [qat.yaml](../lpot/template/qat.yaml) or [pruning.yaml](../lpot/template/pruning.yaml) to work directory and modify correspondingly.

Below is an example for beginner.

```yaml
model:                  # mandatory. lpot uses this model name and framework name to decide where to save tuning history and deploy yaml.
  name: ssd_mobilenet_v1
  framework: tensorflow
  inputs: image_tensor
  outputs: num_detections,detection_boxes,detection_scores,detection_classes

tuning:
  accuracy_criterion:
    relative:  0.01
  exit_policy:
    timeout: 0
    max_trials: 300
  random_seed: 9527
```

Below is an example for advance user, which constrain the tuning space by specifing calibration, quantization, tuning.ops fields accordingly.

```yaml
model:
  name: ssd_mobilenet_v1
  framework: tensorflow
  inputs: image_tensor
  outputs: num_detections,detection_boxes,detection_scores,detection_classes

quantization:
    calibration:
      sampling_size: 10, 50
    model_wise:
      - weight:
          - granularity: per_channel
            scheme: asym
            dtype: int8
            algorithm: minmax
        activation:
          - granularity: per_tensor
            scheme: asym
            dtype: int8
            algorithm: minmax
    op_wise: {
               'conv1': {
                 'activation':  {'dtype': ['uint8', 'fp32'], 'algorithm': ['minmax', 'kl'], 'scheme':['sym']},
                 'weight': {'dtype': ['int8', 'fp32'], 'algorithm': ['kl']}
               }
             }

tuning:
    accuracy_criterion:
      relative:  0.01
    objective: performance
    exit_policy:
      timeout: 36000
      max_trials: 1000
    workspace:
      path: /path/to/saving/directory
      resume: /path/to/a/specified/snapshot/file

```

### 3. Integration with Intel® Low Precision Optimization Tool

   a. Check if calibration or evaluation dataloader in user code meets Intel® Low Precision Optimization Tool requirements, that is whether it returns a tuple of (input, label). In classification networks, its dataloader usually yield output like this. As calication dataset does not need to have label, user need wrapper the loader to return a tuple of (input, _) for Intel® Low Precision Optimization Tool on this case. In object detection or NLP or recommendation networks, its dataloader usually yield output not like this, user need wrapper the loder to return a tuple of (input, label), in which "input" may be a object, a tuple or a dict.

   b. Check if model in user code could be directly feed "input" got from #a. If not, user need wrapper the model to take "input" as input.

   c. If user choose the first use case, that is using Intel® Low Precision Optimization Tool build-in metrics. User need ensure metric built in Intel® Low Precision Optimization Tool could take output of model and label of eval_dataloader as input.


# Model Zoo

|tensorflow|[	resnet50v1.0  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	resnet50v1.5  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	resnet101  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	inception_v1  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	inception_v2  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	inception_v3  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	inception_v4  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	inception_resnet_v2  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	mobilenetv1  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	ssd_resnet50_v1  ](../examples/tensorflow/object_detection/README.md)|
|tensorflow|[	mask_rcnn_inception_v2  ](../examples/tensorflow/object_detection/README.md)|
|tensorflow|[	wide_deep_large_ds  ](../examples/tensorflow/recommendation/wide_deep_large_ds/WND_README.md)|
|tensorflow|[	vgg16  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	vgg19  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	resnetv2_50  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	resnetv2_101  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	resnetv2_152  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	densenet121  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	densenet161  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	densenet169  ](../examples/tensorflow/image_recognition/README.md)|
|tensorflow|[	style_transfer  ](../examples/tensorflow/style_transfer/README.md)|
|tensorflow|[	retinanet  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	googlenet-v3  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet101_kitti  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet101_ava_v2.1  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet101_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	vgg19-oob  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet101_lowproposals_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet50_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	vgg16-oob  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_resnet50_lowproposals_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	rfcn-resnet101-coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	openpose-pose  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	googlenet-v1  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	resnet-50  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	googlenet-v2  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	ssd-resnet34_300x300  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	ssd_resnet50_v1_fpn_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	RetinaNet50  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	googlenet-v4  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	faster_rcnn_inception_v2_coco  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	yolo-v2-ava-sparse-35-0001  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	yolo-v2-ava-sparse-70-0001  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	resnet-152  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	resnet-v2-152  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	resnet-101  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	person-vehicle-bike-detection-crossroad-yolov3-1020  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	squeezenet-1.1  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	yolo-v3  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	resnet-v2-101  ](../examples/tensorflow/oob_models/README.md)|
|tensorflow|[	darknet53  ](../examples/tensorflow/oob_models/README.md)|
|pytorch|[	resnet18  ](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)|
|pytorch|[	resnet50  ](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)|
|pytorch|[	resnext101_32x8d  ](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)|
|pytorch|[	bert_base_MRPC  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_base_CoLA  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_base_STS-B  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_base_SST-2  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_base_RTE  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_large_MRPC  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_large_SQuAD  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_large_QNLI  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_large_RTE  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	bert_large_CoLA  ](../examples/pytorch/language_translation/README.md)|
|pytorch|[	dlrm  ](../examples/pytorch/recommendation/README.md)|
|pytorch|[	resnet18_qat  ](../examples/pytorch/image_recognition/imagenet/cpu/qat/README.md)|
|pytorch|[	resnet50_qat  ](../examples/pytorch/image_recognition/imagenet/cpu/qat/README.md)|
|pytorch|[	inception_v3  ](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)|
|pytorch|[	peleenet  ](../examples/pytorch/image_recognition/peleenet/PeleeNet_README.md)|
|pytorch|[	yolo_v3  ](../examples/pytorch/object_detection/yolo_v3/README.md)|
|pytorch|[	se_resnext50_32x4d  ](../examples/pytorch/image_recognition/se_resnext/README.md)|
|pytorch|[	mobilenet_v2  ](../examples/pytorch/image_recognition/imagenet/cpu/ptq/README.md)|
|pytorch|[	resnest50  ](../examples/pytorch/image_recognition/resnest/README.md)|
|mxnet|[	resnet50v1  ](../examples/mxnet/image_recognition/README.md)|
|mxnet|[	inceptionv3  ](../examples/mxnet/image_recognition/README.md)|
|mxnet|[	mobilenet1.0  ](../examples/mxnet/image_recognition/README.md)|
|mxnet|[	mobilenetv2_1.0  ](../examples/mxnet/image_recognition/README.md)|
|mxnet|[	resnet18_v1  ](../examples/mxnet/image_recognition/README.md)|
|mxnet|[	squeezenet1.0  ](../examples/mxnet/language_translation/README.md)|
|mxnet|[	ssd-resnet50_v1  ](../examples/mxnet/object_detection/README.md)|
|mxnet|[	ssd-mobilenet1.0  ](../examples/mxnet/object_detection/README.md)|
|mxnet|[	resnet152_v1  ](../examples/mxnet/image_recognition/README.md)|

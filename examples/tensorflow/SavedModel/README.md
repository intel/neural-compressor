Step-by-Step
============

This document is used to enable Tensorflow SavedModel format using Intel® Neural Compressor.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow <= 2.4.0.

### 3. Prepare Pretrained model
Download the model from tensorflow-hub.

image recognition
- [mobilenetv1](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v1_075_224/classification/5)
- [resnet50v1](https://hub.tensorflow.google.cn/tensorflow/resnet_50/classification/1)
- [mobilenetv2](https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_035_224/classification/5)
- [efficientnet_v2_b0](https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2)

object detection
- [centernet_resnet50_v1_512](https://hub.tensorflow.google.cn/tensorflow/centernet/resnet50v1_fpn_512x512/1)
- [centernet_resnet50_v2_512](https://hub.tensorflow.google.cn/tensorflow/centernet/resnet50v2_512x512/1)
- [centernet_resnet101_v1_512](https://hub.tensorflow.google.cn/tensorflow/centernet/resnet101v1_fpn_512x512/1)
- [ssd_resnet50_v1(retinaNet)](https://hub.tensorflow.google.cn/tensorflow/retinanet/resnet50_v1_fpn_640x640/1)
- [ssd_mobilenet_v1](https://hub.tensorflow.google.cn/tensorflow/ssd_mobilenet_v1/fpn_640x640/1)
- [faster_rcnn_resnet50_v1](https://hub.tensorflow.google.cn/tensorflow/faster_rcnn/resnet50_v1_640x640/1)


## Run Command
  ```shell
  bash run_tuning.sh --config=./config.yaml --input_model=./SavedModel --output_model=./nc_SavedModel
  ```
  ```shell
  bash run_benchmark.sh --config=./config.yaml --input_model=./SavedModel --mode=performance
  ```
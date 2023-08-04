Step-by-Step
============

This document is used to list steps of reproducing Intel Optimized TensorFlow OOB models tuning zoo result.

# Prerequisite

## 1. Environment

### Installation
  Recommend python 3.8 or higher version.

  ```bash
  # Install Intel® Neural Compressor
  pip install neural-compressor
  # Install Intel® Tensorflow
  pip install intel-tensorflow
  ```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Install Intel Extension for Tensorflow
#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest Intel GPU driver installation.
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers).

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare pre-trained model

  * Get model from [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4/tools/downloader/README.md) 


  ```bash
  git clone https://github.com/openvinotoolkit/open_model_zoo.git
  
  git checkout 2021.4
  
  cd open_model_zoo/tools/downloader/

  ./downloader.py --name ${model_name} --output_dir ${model_path}
  ```
List models names can get with open_model_zoo:

|	Model_name	|
|	--------------------------------	|
|	deeplabv3	|
|	efficientnet-b0	|
|	efficientnet-b0_auto_aug	|
|	efficientnet-b5	|
|	efficientnet-b7_auto_aug	|
|	faster_rcnn_inception_v2_coco	|
|	faster_rcnn_resnet101_coco	|
|	faster_rcnn_resnet50_coco	|
|	googlenet-v1-tf	|
|	googlenet-v3	|
|	googlenet-v4-tf	|
|	i3d-rgb-tf	|
|	inception-resnet-v2-tf	|
|	license-plate-recognition-barrier-0007	|
|	resnet-50-tf	|
|	rfcn-resnet101-coco-tf	|
|	vehicle-license-plate-detection-barrier-0123	|
|	yolo-v2-tf	|
|	yolo-v3-tf	|


* Download with URL

|	Model name	|	URL	|
|	--------------------------------	|	--------------------------------	|
|	faster_rcnn_resnet101_ava_v2.1	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz	|
|	faster_rcnn_resnet101_kitti	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz	|
|	faster_rcnn_resnet101_lowproposals_coco	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz	|
|	image-retrieval-0001	|	https://download.01.org/opencv/openvino_training_extensions/models/image_retrieval/image-retrieval-0001.tar.gz	|
|	SSD ResNet50 V1 FPN 640x640 (RetinaNet50)	|	http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz	|
|	ssd_inception_v2_coco	|	http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz	|
|	ssd-resnet34 300x300	|	https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_fp32_bs1_pretrained_model.pb	|


## 3. Prepare Dataset

  We use dummy data to do benchmarking with Tensorflow OOB models.

# Run
## run tuning

```bash
./run_quant.sh --topology=${model_topology} --dataset_location= --input_model=${model_path} --output_model=${output_model_path}
```

### Quantization Config
The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## run benchmarking

```bash
./run_benchmark.sh --topology=${model_topology} --dataset_location= --input_model=${model_path} --mode=performance --batch_size=1
```

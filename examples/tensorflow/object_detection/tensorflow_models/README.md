Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Object Detection models tuning results. This example can run on Intel CPUs and GPUs.
Currently, we've enabled below models.
 * ssd_resnet50_v1
 * ssd_resnet34
 * ssd_mobilenet_v1
 * fastrcnn_inception_resnet_v2
 * fastrcnn_resnet101
 * fastrcnn_resnet50
 * maskrcnn_inception_v2
## Prerequisite


### 1. Installation
Recommend python 3.6 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../README.md#supported-frameworks).

### 3. Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/tensorflow_models/quantization/ptq
pip install -r requirements.txt
```

### 4. Install Protocol Buffer Compiler

`Protocol Buffer Compiler` in version higher than 3.0.0 is necessary ingredient for automatic COCO dataset preparation. To install please follow
[Protobuf installation instructions](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager).

### 5. Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 6. Prepare Dataset

#### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/tensorflow_models/`.

Usage:
```shell
cd examples/tensorflow/object_detection/tensorflow_models/
. prepare_dataset.sh
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

#### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

### 7. Download Model

#### Automated approach
Run the `prepare_model.py` script located in `examples/tensorflow/object_detection/tensorflow_models/`.

```
usage: prepare_model.py [-h] [--model_name {ssd_resnet50_v1,ssd_mobilenet_v1}]
                        [--model_path MODEL_PATH]

Prepare pre-trained model for COCO object detection

optional arguments:
  -h, --help            show this help message and exit
  --model_name {ssd_resnet50_v1,ssd_mobilenet_v1}
                        model to download, default is ssd_resnet50_v1
  --model_path MODEL_PATH
                        directory to put models, default is ./model
```

#### Manual approach

##### ssd_resnet50_v1
```shell
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
```

##### ssd_mobilenet_V1

```shell
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

##### faster_rcnn_inception_resnet_v2

```shell
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

##### faster_rcnn_resnet101

```shell
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

##### faster_rcnn_resnet50

```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
tar -xvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
```

##### mask_rcnn_inception_v2

```shell
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

##### ssd_resnet34
```shell
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
```
You need to install intel-tensorflow==2.4.0 to enable ssd_resnet34 model.

## Run Command

Now we support both pb, saved_model and ckpt formats.

### For PB model
  
  ```shell
  # The cmd of running ssd_resnet50_v1
  bash run_tuning.sh --input_model=./ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb --output_model=./tensorflow-ssd_resnet50_v1-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

### For ckpt model
  
  ```shell
  # The cmd of running ssd_resnet50_v1
  bash run_tuning.sh --input_model=./ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ --output_model=./tensorflow-ssd_resnet50_v1-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

### For saved_model model
  
  ```shell
  # The cmd of running faster_rcnn_resnet101
  bash run_tuning.sh --input_model=./faster_rcnn_resnet101_coco_2018_01_28/saved_model/ --output_model=./tensorflow-faster_rcnn_resnet101-tune.pb --dataset_location=/path/to/dataset/coco_val.record
  ```

> Note
> For ssd_resnet34 model, anno_path of evaluation/accuracy/metric/COCOmAP in args should be "label_map.yaml"

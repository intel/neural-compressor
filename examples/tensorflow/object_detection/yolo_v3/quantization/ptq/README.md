This document describes the step-by-step to reproduce Yolo-v3 tuning result with Neural Compressor. This example can run on Intel CPUs and GPUs.

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
> Note: Supported Tensorflow versions please refer to Neural Compressor readme file.

### 3. Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
pip install -r requirements.txt
```

### 4. Install Intel Extension for Tensorflow
#### Quantizing the model on Intel CPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel GPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 5. Downloaded Yolo-v3 model
```shell
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
```

### 6. Download COCO Class Names File
```shell
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### 7. Download Model Weights (Full):
```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

### 8. Generate PB:
```shell
python convert_weights_pb.py --class_names coco.names --weights_file yolov3.weights --data_format NHWC --size 416 --output_graph yolov3.pb
```

### 9. Prepare Dataset

#### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/yolo_v3/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
. prepare_dataset.sh
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

#### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

## Get Quantized Yolo-v3 model with Neural Compressor

### 1.Config the yolo_v3.yaml with the valid cocoraw data path or the yolo_v3_itex.yaml if using the Intel Extension for Tensorflow.

### 2.Config the yaml file
In examples directory, there is a yolo_v3.yaml for tuning the model on Intel CPUs. The 'framework' in the yaml is set to 'tensorflow'. If running this example on Intel GPUs, the 'framework' should be set to 'tensorflow_itex' and the device in yaml file should be set to 'gpu'. The yolo_v3_itex.yaml is prepared for the GPU case. We could remove most of items and only keep mandatory item for tuning. We also implement a calibration dataloader and have evaluation field for creation of evaluation function at internal neural_compressor.

### 3.Run below command one by one.
Usage
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
```
```python
python infer_detections.py --input_graph /path/to/yolov3_fp32.pb --config ./yolo_v3.yaml --output_graph /path/to/save/yolov3_tuned3.pb
```

Finally, the program will generate the quantized Yolo-v3 model with relative 1% loss.

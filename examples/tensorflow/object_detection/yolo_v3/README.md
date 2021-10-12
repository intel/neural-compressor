This document describes the step-by-step to reproduce Yolo-v3 tuning result with Neural Compressor.

## Prerequisite


### 1. Installation
Recommend python 3.6 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
Check your python version and pip install 1.15.0 up3 from links as below:   
pip install https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up3-cp36-cp36m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl   
pip install https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
```
> Note: Supported Tensorflow versions please refer to Neural Compressor readme file.

### 3. Installation Dependency packages
```shell
cd examples/tensorflow/object_detection
pip install -r requirements.txt
```

### 4. Downloaded Yolo-v3 model
```shell
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
```

### 5. Download COCO Class Names File
```shell
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### 6. Download Model Weights (Full):
```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

### 7. Generate PB:
```shell
python convert_weights_pb.py --class_names coco.names --weights_file yolov3.weights --data_format NHWC --size 416 --output_graph yolov3.pb
```

### 8. Prepare Dataset

#### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection`.

Usage:
```shell
cd examples/tensorflow/object_detection
. prepare_dataset.sh
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

#### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).

## Get Quantized Yolo-v3 model with Neural Compressor

### 1.Config the yolo_v3.yaml with the valid cocoraw data path.

### 2.Run below command one by one.
Usage
```shell
cd examples/tensorflow/object_detection/yolo_v3
```
```python
python infer_detections.py --input_graph /path/to/yolov3_fp32.pb --config ./yolo_v3.yaml --output_graph /path/to/save/yolov3_tuned3.pb
```

Finally, the program will generate the quantized Yolo-v3 model with relative 1% loss.

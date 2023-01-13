This document describes the step-by-step to reproduce Yolo-v3 tuning result with Neural Compressor. This example can run on Intel CPUs and GPUs.

## Prerequisite


### 1. Environment
Recommend python 3.6 or higher version.

#### 1. Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

#### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```

#### 3. Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
pip install -r requirements.txt
```

#### 4. Install Intel Extension for Tensorflow
##### Quantizing the model on Intel GPU
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel-innersource/frameworks.ai.infrastructure.intel-extension-for-tensorflow.intel-extension-for-tensorflow/blob/master/docs/install/install_for_gpu.md#install-gpu-drivers)

##### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 2. Prepare model
#### 1. Downloaded Yolo-v3 model
```shell
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
```

#### 2. Download COCO Class Names File
```shell
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

#### 3. Download Model Weights (Full):
```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

#### 4. Generate PB:
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


## Run below command one by one.
Usage
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
```
### Tune
```python
bash run_tuning.sh --input_model=/path/to/yolov3_fp32.pb --output_model=/path/to/save/yolov3_int8.pb --dataset_location=/path/to/dataset
```

### Benchmark
```python
bash run_benchmark.sh --input_model=/path/to/yolov3_fp32.pb --dataset_location=/path/to/dataset --mode=performance
```

Finally, the program will generate the quantized Yolo-v3 model with relative 1% loss.

This document describes the step-by-step to reproduce Yolo-v3 tuning result with Neural Compressor. This example can run on Intel CPUs and GPUs.

# Prerequisite


## 1. Environment
Recommend python 3.6 or higher version.

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Installation Dependency packages
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_gpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare model
### Downloaded Yolo-v3 model
```shell
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
```

### Download COCO Class Names File
```shell
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

### Download Model Weights (Full):
```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

### Generate PB:
```shell
python convert_weights_pb.py --class_names coco.names --weights_file yolov3.weights --data_format NHWC --size 416 --output_graph yolov3.pb
```

## 3. Prepare Dataset

### Automatic dataset download

> **_Note: `prepare_dataset.sh` script works with TF version 1.x._**

Run the `prepare_dataset.sh` script located in `examples/tensorflow/object_detection/yolo_v3/quantization/ptq`.

Usage:
```shell
cd examples/tensorflow/object_detection/yolo_v3/quantization/ptq
. prepare_dataset.sh
```

This script will download the *train*, *validation* and *test* COCO datasets. Furthermore it will convert them to
tensorflow records using the `https://github.com/tensorflow/models.git` dedicated script.

### Manual dataset download
Download CoCo Dataset from [Official Website](https://cocodataset.org/#download).


# Run

## Quantization Config

The Quantization Config class has default parameters setting for running on Intel CPUs. If running this example on Intel GPUs, the 'backend' parameter should be set to 'itex' and the 'device' parameter should be set to 'gpu'.

```
config = PostTrainingQuantConfig(
    device="gpu",
    backend="itex",
    ...
    )
```

## 1. Quantization
```python
bash run_quant.sh --input_model=/path/to/yolov3_fp32.pb --output_model=/path/to/save/yolov3_int8.pb --dataset_location=/path/to/dataset
```

## 2. Benchmark
```python
bash run_benchmark.sh --input_model=/path/to/yolov3_fp32.pb --dataset_location=/path/to/dataset --mode=performance
```

Finally, the program will generate the quantized Yolo-v3 model with relative 1% loss.

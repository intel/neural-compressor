This document describes the step-by-step to reproduce Yolo-v5 tuning result with Neural Compressor. This example can run on Intel CPUs and GPUs.

# Prerequisite


## 1. Environment
Recommend python 3.10 or higher version.

### Install Intel® Neural Compressor
```shell
pip install neural-compressor
```

### Install Tensorflow
```shell
pip install tensorflow
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

### Installation Dependency packages
```shell
cd examples/3.x_api/tensorflow/object_detection/yolo_v5/quantization/ptq
pip install -r requirements.txt
```

### Install Intel Extension for Tensorflow

#### Quantizing the model on Intel GPU(Mandatory to install ITEX)
Intel Extension for Tensorflow is mandatory to be installed for quantizing the model on Intel GPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[xpu]
```
For any more details, please follow the procedure in [install-gpu-drivers](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/install/install_for_xpu.md#install-gpu-drivers)

#### Quantizing the model on Intel CPU(Optional to install ITEX)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's not mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

> **Note**: 
> The version compatibility of stock Tensorflow and ITEX can be checked [here](https://github.com/intel/intel-extension-for-tensorflow#compatibility-table). Please make sure you have installed compatible Tensorflow and ITEX.

## 2. Prepare model

Users can choose to automatically or manually download the model.
### Automatic download

Run the `prepare_model.sh` script.
```shell
. prepare_model.sh
```

This script will load yolov5 model to `./yolov5/yolov5s.pb`.

### Manual download

To get a TensorFlow pretrained model, you need to export it from a PyTorch model. Clone the [Ultralytics yolov5 repository](https://github.com/ultralytics/yolov5.git).
Generate the pretrained PyTorch model and then export to a Tensorflow supported format with the following commands:
```shell
python yolov5/models/tf.py --weights yolov5/yolov5s.pt
python yolov5/export.py --weights yolov5/yolov5s.pt --include pb
```

The yolov5 model will be loaded to `./yolov5/yolov5s.pb`.

## 3. Prepare Dataset

Users can choose to automatically or manually download the dataset.
### Automatic download

Run the `prepare_dataset.sh` script.
```shell
. prepare_dataset.sh
```
The validation set of coco2017 will be downloaded into a `./coco` folder.

# Run

## 1. Quantization
```python
bash run_quant.sh --input_model=./yolov5/yolov5s.pb --output_model=yolov5s_int8.pb --dataset_location=/path/to/dataset
```

## 2. Benchmark
```python
# run performance benchmark
bash run_benchmark.sh --input_model=yolov5s_int8.pb --dataset_location=/path/to/dataset --mode=performance

# run accuracy benchmark
bash run_benchmark.sh --input_model=yolov5s_int8.pb --dataset_location=/path/to/dataset --mode=accuracy
```

Finally, the program will generate the quantized Yolo-v5 model with relative 1% loss.

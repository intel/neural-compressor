Step-by-Step
============

This example load an object detection model converted from Tensorflow and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). 

# Prerequisite

## 1. Environment

```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model
Please refer to [Converting SSDMobilenet To ONNX Tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for detailed model converted. The following is a simple example command:

```shell
export MODEL=ssd_mobilenet_v1_coco_2018_01_28
wget http://download.tensorflow.org/models/object_detection/$MODEL.tar.gz
tar -xvf $MODEL.tar.gz

python -m tf2onnx.convert --graphdef $MODEL/frozen_inference_graph.pb --output ./$MODEL.onnx --fold_const --opset 11 --inputs image_tensor:0 --outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0
```

## 3. Prepare Dataset

Download [MS COCO 2017 dataset](https://cocodataset.org/#download).

# Run

## Diagnosis
Neural Compressor offers quantization and benchmark diagnosis. Adding `diagnosis` parameter to Quantization/Benchmark config will provide additional details useful in diagnostics.
### Quantization diagnosis
```
config = PostTrainingQuantConfig(
    diagnosis=True,
    ...
)
``` 

### Benchmark diagnosis
```
config = BenchmarkConfig(
    diagnosis=True,
    ...
)
``` 

## 1. Quantization

Static quantization with QOperator format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \
                   --quant_format="QOperator"
```

Static quantization with QDQ format:

```bash
bash run_quant.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \ # model path as *.onnx
                   --dataset_location=path/to/val2017 \
                   --quant_format="QDQ"
```

## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=path/to/val2017 \
                      --mode=performance # or accuracy
```

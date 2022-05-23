# Evaluate performance of ONNX Runtime(SSD Mobilenet v1) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model
Please refer to [Converting SSDMobilenet To ONNX Tutorial](https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for detailed model converted. The following is a simple example command:

```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=ssd_mobilenet_v1.yaml \ 
                   --output_model=path/to/save
```
Make sure **anno_path** in ssd_mobilenet_v1.yaml is the path of label_map.yaml.

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=ssd_mobilenet_v1_qdq.yaml \ 
                   --output_model=path/to/save
```
Make sure **anno_path** in ssd_mobilenet_v1_qdq.yaml is the path of label_map.yaml.

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=ssd.yaml \
                      --mode=performance # or accuracy

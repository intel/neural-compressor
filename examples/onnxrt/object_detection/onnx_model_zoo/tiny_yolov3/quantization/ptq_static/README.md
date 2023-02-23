# Evaluate performance of ONNX Runtime(Tiny-YOLOv3) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.11.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --output_model=path/to/save \
                   --dataset_location=path/to/coco/dataset \
                   --quant_format="QOperator"
```

### Performance

```bash
bash run_benchmark.sh --input_model=path/to/model  \ # model path as *.onnx
                      --dataset_location=path/to/coco/dataset \
                      --mode=performance # or accuracy
```


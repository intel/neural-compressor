# Evaluate performance of ONNX Runtime(FCN) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.12.0  
onnxruntime: 1.13.1
> Validated framework versions can be found in main readme.

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --dataset_location=path/to/coco/val2017 \
                   --output_model=path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --dataset_location=path/to/coco/val2017 \
                   --output_model=path/to/save \
                   --quant_format=QDQ
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=path/to/coco/val2017 \
                      --mode=performance # or accuracy
```

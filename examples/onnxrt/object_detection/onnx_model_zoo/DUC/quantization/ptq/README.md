# Evaluate performance of ONNX Runtime(ResNet101-DUC) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [cityscapes dataset](https://www.cityscapes-dataset.com/downloads/). You need to download this dataset yourself.

### Environment
onnx: 1.12.0  
onnxruntime: 1.13.1
> Validated framework versions can be found in main readme.

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell

wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --dataset_location=/path/to/leftImg8bit/val \
                   --output_model=path/to/save
```

### Performance

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/leftImg8bit/val \
                      --mode=performance
```

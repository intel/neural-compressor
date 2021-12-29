# Evaluate performance of ONNX Runtime(FCN) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an object detection model converted from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [MS COCO 2017 dataset](https://cocodataset.org/#download). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.8.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model  \ # model path as *.onnx
                   --config=fcn_rn50.yaml \ 
                   --data_path=path/to/coco/val2017 \
                   --label_path=path/to/coco/annotations/instances_val2017.json \
                   --output_model=path/to/save
```

### Performance

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=fcn_rn50.yaml \
                      --data_path=path/to/coco/val2017 \
                      --label_path=path/to/coco/annotations/instances_val2017.json \
                      --mode=performance
```

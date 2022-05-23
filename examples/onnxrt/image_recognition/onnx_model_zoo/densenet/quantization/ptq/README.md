# Evaluate performance of ONNX Runtime(Densenet) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/classification/densenet-121/model/densenet-12.onnx
```

### Quantization

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=densenet.yaml \
                   --output_model=path/to/save
```

### Performance 

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=densenet.yaml \
                      --mode=performance
```


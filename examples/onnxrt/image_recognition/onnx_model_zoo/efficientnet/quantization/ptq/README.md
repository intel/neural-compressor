# Evaluate performance of ONNX Runtime(EfficientNet-Lite4) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model from [ONNX Model Zoo](https://github.com/onnx/models) and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model
Download model from [ONNX Model Zoo](https://github.com/onnx/models)

```shell
wget https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=efficientnet.yaml \
                   --output_model=path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=efficientnet_qdq.yaml \
                   --output_model=path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=efficientnet.yaml \
                      --mode=performance # or accuracy
```


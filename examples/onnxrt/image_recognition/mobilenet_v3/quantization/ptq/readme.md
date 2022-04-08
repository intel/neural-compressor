# Evaluate performance of ONNX Runtime(Mobilenet v3) 
>ONNX runtime quantization is under active development. please use 1.6.0+ to get more quantization support. 

This example load an image classification model exported from PyTorch and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

### Environment
onnx: 1.9.0
onnxruntime: 1.10.0

### Prepare model
Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.

```bash
wget https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite

python -m tf2onnx.convert --opset 11 --tflite mobilenet_edgetpu_224_1.0_float.tflite --output mobilenet_v3.onnx
```

### Quantization

Quantize model with QLinearOps:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mobilenet_v3.yaml \ 
                   --output_model=path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mobilenet_v3_qdq.yaml \ 
                   --output_model=path/to/save
```

### Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --config=mobilenet_v3.yaml \
                      --mode=performance # or accuracy
```

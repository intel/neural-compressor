Step-by-Step
============

This example load an image classification model exported from PyTorch and confirm its accuracy and speed based on [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads). You need to download this dataset yourself.

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model
Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.

```bash
wget https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_float.tflite

python -m tf2onnx.convert --opset 11 --tflite mobilenet_edgetpu_224_1.0_float.tflite --output mobilenet_v3.onnx
```

## 3. Prepare Dataset
Download dataset [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads).

Download label:

```shell
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xvzf caffe_ilsvrc12.tar.gz val.txt
```

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

Quantize model with QLinearOps:

```bash
bash run_quant.sh --input_model=path/to/model \  # model path as *.onnx
                   --dataset_location=/path/to/imagenet \
                   --label_path=/path/to/val.txt \
                   --output_model=path/to/save
```

Quantize model with QDQ mode:

```bash
bash run_quant.sh --input_model=path/to/model \  # model path as *.onnx
                   --dataset_location=/path/to/imagenet \
                   --label_path=/path/to/val.txt \
                   --output_model=path/to/save \
                   --quant_format=QDQ
```


## 2. Benchmark

```bash
bash run_benchmark.sh --input_model=path/to/model \  # model path as *.onnx
                      --dataset_location=/path/to/imagenet \
                      --label_path=/path/to/val.txt \
                      --mode=performance # or accuracy
```
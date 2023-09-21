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
Please refer to [pytorch official guide](https://pytorch.org/docs/stable/onnx.html) for detailed model export. The following is a simple example:

```shell
python prepare_model.py --output_model='vgg16.onnx'
```

## 3. Prepare Dataset
Download dataset [ILSVR2012 validation Imagenet dataset](http://www.image-net.org/challenges/LSVRC/2012/downloads).

Download label:

```shell
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xvzf caffe_ilsvrc12.tar.gz val.txt
```

# Run

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
                      --batch_size=batch_size \
                      --mode=performance # or accuracy
```
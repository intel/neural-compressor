Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch se_resnext tuning and benchmarking results with Intel® Neural Compressor.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/image_recognition/se_resnext/quantization/ptq/fx
pip install -r requirements.txt
```
## 2. Install from Repo
```shell
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd pretrained-models.pytorch
git checkout 8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0
python setup.py install
cd ..
```
> **Note**
>
> Please don't install public pretrainedmodels package.
## 3. Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir should include below folder:
```bash
ls /path/to/imagenet
train  val
```

# Run
## 1. Quantization
```shell
python run_eval.py \
          --data /path/to/imagenet \
          -a se_resnext50_32x4d \
          -b 128 \
          -j 1 \
          -t
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --int8=true --config=saved_results --mode=performance --input_model=se_resnext50_32x4d  --dataset_location=/path/to/imagenet
# fp32
sh run_benchmark.sh --mode=performance --input_model=se_resnext50_32x4d  --dataset_location=/path/to/imagenet
```

# Original SE_ResNext README
Please refer [SE_ResNext README](https://github.com/hujie-frank/SENet/blob/master/README.md).

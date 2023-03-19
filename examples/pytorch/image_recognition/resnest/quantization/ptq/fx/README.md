Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch ResNest50 tuning and benchmarking results with Intel® Neural Compressor.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/image_recognition/resnest/quantization/ptq/fx
pip install -r requirements.txt
```
## 2. Prepare Dataset
Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir should include below folder:
```shell
ls /path/to/imagenet
train  val
```
## 3. Pytorch Models
### GitHub Install
```shell
git clone https://github.com/zhanghang1989/ResNeSt.git
cd ResNeSt
git checkout 1dfb3e8867e2ece1c28a65c9db1cded2818a2031
python setup.py install
cd ..
```
### Load Models
- Load using Torch Hub
```python
import torch
# get list of models and save to cache
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
# load pretrained models, using ResNeSt-50 as an example
net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
```
or

- Load using python package
```python
# using ResNeSt-50 as an example
from resnest.torch import resnest50
net = resnest50(pretrained=True)
```

# Run
## 1. Quantization
```Shell
python -u verify.py --tune --model resnest50 --batch-size 30 --workers 1 --no-cuda /path/to/imagenet
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --int8=true --mode=performance --input_model=resnest50  --dataset_location=/path/to/imagenet --config=saved_results
# fp32
sh run_benchmark.sh --mode=performance --input_model=resnest50  --dataset_location=/path/to/imagenet
```

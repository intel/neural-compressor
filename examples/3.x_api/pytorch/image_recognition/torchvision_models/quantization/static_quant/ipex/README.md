Step-by-Step
============

This document describes the step-by-step instructions for reproducing PyTorch tuning results with Intel® Neural Compressor.

# Prerequisite

## 1. Environment

We verified examples with IPEX backend on Python 3.10, recommended.

```shell
pip install -r requirements.txt
```

## 2. Install Intel-Pytorch-Extension

Please refer to [intel/intel-extension-for-pytorch(github.com)](https://github.com/intel/intel-extension-for-pytorch).

### Install IPEX CPU

   > Note: GCC9 compiler is recommended

   ```shell
   python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
   ```

### Install IPEX XPU
Please build an IPEX docker container according to the [official guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu&os=linux%2fwsl2&package=docker).

You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system. The command should return PyTorch and IPEX versions installed, as well as GPU card(s) information detected.
```bash
source {DPCPPROOT}/env/vars.sh
source {MKLROOT}/env/vars.sh
source {CCLROOT}/env/vars.sh
source {MPIROOT}/env/vars.sh
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```
Please also refer to this [tutorial](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu&os=linux%2fwsl2&package=conda) to check system requirements and install dependencies.

## 3. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet. The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run with CPU

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. ResNet18 With Intel PyTorch Extension

```shell
python main.py -t -a resnet18 --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnet18 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnet18 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 2. ResNet50 With Intel PyTorch Extension

```shell
python main.py -t -a resnet50 --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnet50 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnet50 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

### 3. ResNext101_32x16d With Intel PyTorch Extension

```shell
python main.py -t -a resnext101_32x16d_wsl --hub --ipex --pretrained /path/to/imagenet
```
or
```shell
bash run_quant.sh --input_model=resnext101_32x16d_wsl --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnext101_32x16d_wsl --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false
```

# Run with XPU

> Note: All torchvision model names can be passed as long as they are included in `torchvision.models`, below are some examples.

### 1. ResNet18 With Intel PyTorch Extension

```shell
python main.py -t -a resnet18 --ipex --pretrained /path/to/imagenet --xpu
```
or
```shell
bash run_quant.sh --input_model=resnet18 --dataset_location=/path/to/imagenet
bash run_benchmark.sh --input_model=resnet18 --dataset_location=/path/to/imagenet --mode=performance/accuracy --int8=true/false --xpu=true/false
```

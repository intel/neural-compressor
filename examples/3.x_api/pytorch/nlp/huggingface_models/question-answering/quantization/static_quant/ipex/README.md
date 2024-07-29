Step-by-Step
============
This document describes the step-by-step instructions for reproducing Huggingface models with IPEX backend tuning results with Intel® Neural Compressor.
> Note: IPEX version >= 1.10

# Prerequisite

## 1. Environment
Recommend python 3.6 or higher version.
```shell
pip install -r requirements.txt
pip install torch
pip install intel_extension_for_pytorch
```

# Quantization

## 1. Quantization with CPU
If IPEX version is equal or higher than 1.12, please install transformers 4.19.0.
```shell
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --no_cuda \
    --tune \
    --output_dir ./savedresult
```

## 2. Quantization with XPU
### 2.1 Environment Setting
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

#### 2.2 Quantization Command
```shell
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --xpu \
    --tune \
    --output_dir ./savedresult
```

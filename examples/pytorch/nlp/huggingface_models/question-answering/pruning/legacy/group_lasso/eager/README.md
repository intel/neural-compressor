Step-by-Step
============

This document describes the steps of training with sparsity for BERT model on PyTorch.

# Prerequisite

### 1. Installation

#### Python First

Recommend python 3.7 or higher version.

#### Install [neural_compressor](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot#installation)

```shell
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py install
```

#### Install PyTorch

Install pytorch-gpu, visit [pytorch.org](https://pytorch.org/).
```bash
# Install pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Install BERT dependency

```bash
cd examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager/
pip3 install -r requirements.txt --ignore-installed PyYAML
```

### 1. Prepare Dataset
* For SQuAD task, you should download SQuAD dataset from [SQuAD dataset link](https://rajpurkar.github.io/SQuAD-explorer/).
* Considering SQuAD1.1 Dataset as the example, simply run the following commands
```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```
* Please also ensure that official evaluation script is downloaded to the same directory as json files. vocab.txt is also required for tokenization
* After all data is settled, the data directory should be like this:

### 2. Prepare pretrained model
* Please download BERT large pretrained model from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_pretraining_amp_lamb/files?version=20.03.0).
```bash
# wget cmd
wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/20.03.0/files/bert_large_pretrained_amp.pt

# curl cmd
curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/20.03.0/files/bert_large_pretrained_amp.pt
```
### 3. Run
Enter your created conda env, then run the script. Below command shows how to train with sparsity for a simplified BERT with [one sparse GEMM layer](prune_bert.yaml). 
```bash
bash scripts/run_squad_sparse.sh /path/to/model.pt 2.0 16 5e-5 tf32 /path/to/data /path/to/outdir prune_bert.yaml
```
The default parameters are as follows:
```shell
init_checkpoint=${1:-"/path/to/ckpt_8601.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"4"}
learning_rate=${4:-"3e-5"}
precision=${5:-"tf32"}
BERT_PREP_WORKING_DIR=${6:-'/path/to/bert_data'}
OUT_DIR=${7:-"./results/SQuAD"}
prune_config=${8:-"prune_bert.yaml"}
```
We also provided an example of BERT with [full sparse layers](prune_all.yaml).

# Original BERT README

Please refer [BERT README](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/README.md)

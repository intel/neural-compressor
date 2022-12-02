Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.

# Prerequisite

## 1. Installation

The dependent packages are all in requirements, please install as following.

```
pip install -r requirements.txt
```

## 2. Run

If the automatic download from modelhub fails, you can download [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B?text=My+name+is+Clara+and+I+am) offline.

```shell

python run_clm.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --tune \
  --output_dir /path/to/checkpoint/dir
```


## 3. Command

```
bash run_tuning.sh --topology=gpt_j_wikitext
bash run_benchmark.sh --topology=gpt_j_wikitext --mode=performance --int8=true
```

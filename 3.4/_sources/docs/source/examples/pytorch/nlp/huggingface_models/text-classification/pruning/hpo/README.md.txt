Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface models with HPO feature using the Intel® Neural Compressor.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in `requirements.txt`, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/pruning/hpo/
pip install -r requirements.txt
```
## 2. Prepare Dataset

The dataset will be downloaded automatically from the datasets Hub.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)

# Run
To get tuned model and its accuracy: 
```shell
python run_glue_no_trainer.py \
        --model_name_or_path M-FAC/bert-mini-finetuned-mrpc \
        --task_name mrpc \
        --per_device_eval_batch_size 18 \
        --per_device_train_batch_size 18 \
        --do_prune

```
Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT pattern lock pruning examples result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/text-classification/README.md).

# Prerequisite

## Python Version

Recommend python 3.6 or higher version.


## Install dependency

```shell
pip install -r requirements.txt
```

# Start to neural_compressor tune for Model Pruning

Below are example NLP tasks for model pruning together with task specific fine-tuning.
It requires the pre-trained bert-base sparsity model `Intel/bert-base-uncased-sparse-70-unstructured` from Intel Huggingface portal.
The pruning configuration is specified in yaml file i.e. prune.yaml.

## MNLI task

```bash
python run_glue_no_trainer_prune.py --task_name mnli --max_length 128
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured
       --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```

## SST-2 task

```bash
python run_glue_no_trainer_prune.py --task_name sst2 --max_length 128
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured
       --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```

## QQP task

```bash
python run_glue_no_trainer_prune.py --task_name qqp --max_length 128
       --model_name_or_path Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier
       --seed 42 --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3
       --weight_decay 0.1 --num_warmup_steps 1700 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt
```

## QNLI task

```bash
python run_glue_no_trainer_prune.py --task_name qnli --max_length 128
       --model_name_or_path Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier
       --seed 42 --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3
       --weight_decay 0.1 --num_warmup_steps 500 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt
```

Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT gradient sensitivity pruning examples result.
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
An example yaml `head_conf.yaml` is provided to support the gradient sensitivity pruning algorithm from https://arxiv.org/abs/2010.13382. 

## SST-2 task

```bash
python run_glue_no_trainer_gradient_prune.py
      --task_name sst2 --max_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3
      --seed 5143 --model_name_or_path bert-base-uncased --config ./head_conf.yaml
      --do_prune --do_eval --output_model /path/to/output/
```
Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT pattern lock pruning examples result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/question-answering/README.md).

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

## SQuAD task

```bash
python run_qa_no_trainer_prune.py
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured --dataset_name squad
       --max_seq_length 384 --doc_stride 128 --weight_decay 0.01 --num_warmup_steps 900 --learning_rate 1e-4
       --num_train_epochs 2 --per_device_train_batch_size 12 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```

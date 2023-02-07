Step-by-Step
============

This document is used to list steps of reproducing Huggingface models distillation examples result.

# Prerequisite

## Environment
Recommend python 3.6 or higher version.
```shell
pip install -r requirements.txt
```

# Distillation
## SST-2 task

```bash
wget -P ./ http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.50d.txt -d ./
python run_glue_no_trainer_distillation.py \
      --task_name sst2 --max_seq_length 128 --model_name_or_path BiLSTM \
      --teacher_model_name_or_path textattack/roberta-base-SST-2 --do_distillation \
      --per_device_train_batch_size 32 --learning_rate 1e-4 --num_train_epochs 20 \
      --output_dir /path/to/output_dir  \
      --augmented_sst2_data --seed 5143
```

## MNLI task

```bash
python run_glue_no_trainer_distillation.py \
      --task_name mnli --model_name_or_path huawei-noah/TinyBERT_General_4L_312D \
      --teacher_model_name_or_path blackbird/bert-base-uncased-MNLI-v1 --do_distillation \
      --learning_rate 2e-5 --num_train_epochs 4 --per_device_train_batch_size 32 \
      --output_dir /path/to/output_dir --loss_weights 0 1 --temperature 4 --seed 5143
```

## QQP task

```bash
python run_glue_no_trainer_distillation.py \
      --task_name qqp --max_seq_length 128 --model_name_or_path nreimers/MiniLM-L3-H384-uncased \
      --teacher_model_name_or_path textattack/bert-base-uncased-QQP --do_distillation \
      --per_device_train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 10 \
      --output_dir /path/to/output_dir  --loss_weights 0 1 \
      --temperature 2 --seed 5143
```

## COLA task

```bash
python run_glue_no_trainer_distillation.py \
      --task_name cola --max_seq_length 128 --model_name_or_path distilroberta-base \
      --teacher_model_name_or_path howey/roberta-large-cola --do_distillation \
      --per_device_train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 10 \
      --output_dir /path/to/output_dir  --temperature 2 --seed 5143
```


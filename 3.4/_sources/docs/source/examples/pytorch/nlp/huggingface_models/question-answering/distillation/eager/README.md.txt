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
```bash
python run_qa_no_trainer_distillation.py \
      --dataset_name squad --model_name_or_path distilbert-base-uncased \
      --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 --do_distillation \
      --learning_rate 1e-5 --num_train_epochs 4 --output_dir /path/to/output_dir \
      --loss_weights 0 1 --temperature 2 --run_teacher_logits \
      --pad_to_max_length --seed 5143
```

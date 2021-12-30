Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT distillation examples result.
Original BERT documents please refer to [BERT README](../../../common/README.md) and [README](../../../common/examples/question-answering/README.md).

# Prerequisite

## Python Version

Recommend python 3.6 or higher version.


## Install dependency

```shell
pip install -r requirements.txt
```

# Start to neural_compressor tune for Model Distillation

Below are example NLP tasks for model distillation from a task specific fine-tuned large model to a smaller model.
It requires the pre-trained task specific model such as `csarron/bert-base-uncased-squad-v1` from Huggingface portal.
The distillation configuration is specified in yaml file i.e. distillation.yaml.

## SQuAD task

```bash
python run_qa_no_trainer_distillation.py \
      --dataset_name squad --model_name_or_path distilbert-base-uncased \
      --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 --do_distillation \
      --learning_rate 1e-5 --num_train_epochs 4 --output_dir /path/to/output_dir \
      --loss_weights 0 1 --temperature 2 --run_teacher_logits \
      --pad_to_max_length --seed 5143
```

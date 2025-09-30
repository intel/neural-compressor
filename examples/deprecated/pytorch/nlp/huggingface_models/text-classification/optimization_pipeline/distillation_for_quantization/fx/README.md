Step-by-Step
============

This document is used to illustrate how to run the distillation for quantization examples.
<br>
These examples will take a NLP model fine tuned on the down stream task, use its copy as a teacher model, and do distillation during the process of quantization aware training.
<br>
For more information of this algorithm, please refer to the paper [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)

# Prerequisite

## Environment

Recommend python 3.7 or higher version.

```shell
pip install -r requirements.txt
```

# Run

## SST-2 task

```bash
python run_glue_no_trainer.py --task_name sst2 --model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2 --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-sst2 --batch_size 32 --do_eval --do_quantization --do_distillation --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## MNLI task

```bash
python run_glue_no_trainer.py --task_name mnli --model_name_or_path yoshitomo-matsubara/bert-base-uncased-mnli --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-mnli --batch_size 32 --do_eval --do_quantization --do_distillation --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## QQP task

```bash
python run_glue_no_trainer.py --task_name qqp --model_name_or_path yoshitomo-matsubara/bert-base-uncased-qqp --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-qqp --batch_size 32 --do_eval --do_quantization --do_distillation --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

## QNLI task

```bash
python run_glue_no_trainer.py --task_name qnli --model_name_or_path yoshitomo-matsubara/bert-base-uncased-qnli --teacher_model_name_or_path yoshitomo-matsubara/bert-base-uncased-qnli --batch_size 32 --do_eval --do_quantization --do_distillation --pad_to_max_length --num_train_epochs 9 --output_dir /path/to/output_dir
```

# Results
We listed the results on 4 distillation for quantization experiments, for comparison, we also listed the results of QAT as well as the baselie metrics of the FP32 model. These experiments use a fine-tuned BERT-Base model on 4 GLUE tasks (SST-2, QNLI, QQP and MNLI), data in the column of FP32 is the metrics of the 4 fine-tuned BERT-Base model, data in the column of INT8 (QAT) is the metrics of the 4 INT8 BERT-Base models from QAT process, data in the column of INT8 (Distillation for Quantization) is the metrics of the 4 INT8 BERT-Base models from distillation for quantization process.
  |               |    FP32        |           INT8 (QAT)     |  INT8 (Distillation for Quantization) |
  |---------------|----------------|--------------------------|--------------------------|
  |  SST-2 (ACC)  |     92.48%     |         91.90%           |          92.01%          |
  |  QNLI (ACC)   |     91.58%     |         89.49%           |          90.33%          |
  |  QQP (ACC/F1) | 90.95%/87.83%  |       89.60%/86.56%      |        91.07%/87.91%     |
  |  MNLI (ACC)   |     84.20%     |         78.67%           |          84.42%          |

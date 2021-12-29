Step-by-Step
============

This document is used to list steps of reproducing Prune Once For All examples result.
<br>
These examples will take the pre-trained sparse language model and fine tune it on the several downstream tasks. This fine tune pipeline is two staged. For stage 1, the pattern lock pruning and the distillation are applied to fine tune the pre-trained sparse language model. In stage 2, the pattern lock pruning, distillation and quantization aware training are performed simultaneously on the fine tuned model from stage 1 to obtain the quantized model with the same sparsity pattern as the pre-trained sparse language model.
<br>
For more informations of this algorithm, please refer to the paper [Prune Once For All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

# Prerequisite

## Python Version

Recommend python 3.6 or higher version.


## Install dependency

```shell
pip install -r requirements.txt
```

# Start running neural_compressor implementation of Prune Once For All

Below are example NLP tasks for Prune Once For All to fine tune the sparse BERT model on the specific task.
<br>
It requires the pre-trained task specific model such as `csarron/bert-base-uncased-squad-v1` from Huggingface portal as the teacher model for distillation, also the pre-trained sparse BERT model such as `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa` from Intel Huggingface portal as the model for fine tuning.
<br>
The pattern lock pruning configuration is specified in yaml file i.e. prune.yaml, the quantization aware training configuration is specified in yaml file i.e. qat.yaml.

## SQuAD task

```bash
# for stage 1
python run_qa_no_trainer_pruneOFA.py --dataset_name squad \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 \
      --do_prune --do_distillation --max_seq_length 384 --batch_size 12 \
      --learning_rate 1.5e-4 --do_eval --num_train_epochs 8 \
      --output_dir /path/to/stage1_output_dir --loss_weights 0 1 \
      --temperature 2 --seed 5143 --pad_to_max_length --run_teacher_logits
# for stage 2
python run_qa_no_trainer_pruneOFA.py --dataset_name squad \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 \
      --do_prune --do_distillation --max_seq_length 384 --batch_size 12 \
      --learning_rate 1e-5 --do_eval --num_train_epochs 2 --do_quantization \
      --output_dir /path/to/stage2_output_dir --loss_weights 0 1 \
      --temperature 2 --seed 5143 --pad_to_max_length  --run_teacher_logits \
      --resume /path/to/stage1_output_dir/best_model_weights.pt
```
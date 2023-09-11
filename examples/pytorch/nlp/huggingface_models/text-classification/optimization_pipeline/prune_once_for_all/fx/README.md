Step-by-Step
============

This document is used to list steps of reproducing Prune Once For All examples result.
<br>
These examples take the pre-trained sparse language model and fine tune it on several downstream tasks. This fine tune pipeline is two staged. For stage 1, the pattern lock pruning and the distillation are applied to fine-tune the pre-trained sparse language model. In stage 2, the pattern lock pruning, distillation and quantization aware training are performed simultaneously on the fine tuned model from stage 1 to obtain the quantized model with the same sparsity pattern as the pre-trained sparse language model.
<br>
For more information of this algorithm, please refer to the paper [Prune Once For All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

# Prerequisite

## Environment

Recommend python 3.6 or higher version.

```shell
pip install -r requirements.txt
```

# Run

## SST-2 task

```bash
# for stage 1
python run_glue_no_trainer_pruneOFA.py --task_name sst2 \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-SST-2 \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir /path/to/stage1_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143
# for stage 2
python run_glue_no_trainer_pruneOFA.py --task_name sst2 \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-SST-2 \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-5 --num_train_epochs 3 --output_dir /path/to/stage2_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143 --do_quantization \
      --resume /path/to/stage1_output_dir/best_model.pt --pad_to_max_length
```

## MNLI task

```bash
# for stage 1
python run_glue_no_trainer_pruneOFA.py --task_name mnli \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path blackbird/bert-base-uncased-MNLI-v1 \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir /path/to/stage1_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143
# for stage 2
python run_glue_no_trainer_pruneOFA.py --task_name mnli \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path blackbird/bert-base-uncased-MNLI-v1 \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-5 --num_train_epochs 3 --output_dir /path/to/stage2_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143 --do_quantization \
      --resume /path/to/stage1_output_dir/best_model.pt --pad_to_max_length
```

## QQP task

```bash
# for stage 1
python run_glue_no_trainer_pruneOFA.py --task_name qqp \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-QQP \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir /path/to/stage1_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143
# for stage 2
python run_glue_no_trainer_pruneOFA.py --task_name qqp \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-QQP \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-5 --num_train_epochs 3 --output_dir /path/to/stage2_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143 --do_quantization \
      --resume /path/to/stage1_output_dir/best_model.pt --pad_to_max_length
```

## QNLI task

```bash
# for stage 1
python run_glue_no_trainer_pruneOFA.py --task_name qnli \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-QNLI \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir /path/to/stage1_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143
# for stage 2
python run_glue_no_trainer_pruneOFA.py --task_name qnli \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-QNLI \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-5 --num_train_epochs 3 --output_dir /path/to/stage2_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143 --do_quantization \
      --resume /path/to/stage1_output_dir/best_model.pt --pad_to_max_length
```

We supporte Distributed Data Parallel training on single node and multi nodes settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command of stage 1 for SST2 task will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please aware that using CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi-nodes setting, the following command needs to be lanuched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
python -m torch.distributed.launch --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
      run_glue_no_trainer_pruneOFA.py --task_name sst2 \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path textattack/bert-base-uncased-SST-2 \
      --do_prune --do_distillation --max_seq_length 128 --batch_size 32 \
      --learning_rate 1e-4 --num_train_epochs 9 --output_dir /path/to/stage1_output_dir \
      --loss_weights 0 1 --temperature 2 --seed 5143
```

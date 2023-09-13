Step-by-Step
============

This document is used to list steps of reproducing Prune Once For All examples result.
<br>
These examples will take the pre-trained sparse language model and fine tune it on the several downstream tasks. This fine tune pipeline is two staged. For stage 1, the pattern lock pruning and the distillation are applied to fine tune the pre-trained sparse language model. In stage 2, the pattern lock pruning, distillation and quantization aware training are performed simultaneously on the fine tuned model from stage 1 to obtain the quantized model with the same sparsity pattern as the pre-trained sparse language model.
<br>
For more information of this algorithm, please refer to the paper [Prune Once For All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

# Prerequisite

## Environment
Recommend python 3.6 or higher version.

```shell
pip install -r requirements.txt
```

# Prune Once For All

Below are example NLP tasks for Prune Once For All to fine tune the sparse BERT model on the specific task.
<br>
It requires the pre-trained task specific model such as `csarron/bert-base-uncased-squad-v1` from Huggingface portal as the teacher model for distillation, also the pre-trained sparse BERT model such as `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa` from Intel Huggingface portal as the model for fine tuning.
<br>
The pattern lock pruning configuration is specified in yaml file i.e. prune.yaml, the quantization aware training configuration is specified in yaml file i.e. qat.yaml.

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
      --resume /path/to/stage1_output_dir/best_model.pt
```

## Distributed Data Parallel Training

We also supported Distributed Data Parallel training on single node and multi nodes settings. To use Distributed Data Parallel to speedup training, the bash command needs a small adjustment.
<br>
For example, bash command of stage 1 for SQuAD task will look like the following, where *`<MASTER_ADDRESS>`* is the address of the master node, it won't be necessary for single node case, *`<NUM_PROCESSES_PER_NODE>`* is the desired processes to use in current node, for node with GPU, usually set to number of GPUs in this node, for node without GPU and use CPU for training, it's recommended set to 1, *`<NUM_NODES>`* is the number of nodes to use, *`<NODE_RANK>`* is the rank of the current node, rank starts from 0 to *`<NUM_NODES>`*`-1`.
<br>
Also please note that to use CPU for training in each node with multi nodes settings, argument `--no_cuda` is mandatory. In multi nodes setting, following command needs to be launched in each node, and all the commands should be the same except for *`<NODE_RANK>`*, which should be integer from 0 to *`<NUM_NODES>`*`-1` assigned to each node.

```bash
torchrun --master_addr=<MASTER_ADDRESS> --nproc_per_node=<NUM_PROCESSES_PER_NODE> --nnodes=<NUM_NODES> --node_rank=<NODE_RANK> \
      run_qa_no_trainer_pruneOFA.py --dataset_name squad \
      --model_name_or_path Intel/bert-base-uncased-sparse-90-unstructured-pruneofa \
      --teacher_model_name_or_path csarron/bert-base-uncased-squad-v1 \
      --do_prune --do_distillation --max_seq_length 384 --batch_size 12 \
      --learning_rate 1.5e-4 --do_eval --num_train_epochs 8 \
      --output_dir /path/to/stage1_output_dir --loss_weights 0 1 \
      --temperature 2 --seed 5143 --pad_to_max_length --run_teacher_logits
```

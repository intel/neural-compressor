## Examples
we have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.
### [SQuAD](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/question-answering/pruning)

We can train a Squad sparse model with N:M(2:4) pattern:
```shell
python3 ./run_qa_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_bertmini/" \
        --do_prune \
        --target_sparsity 0.5 \
        --pruning_pattern "2:4" \
        --pruning_frequency 1000 \
        --cooldown_epochs 5 \
        --learning_rate 4.5e-4 \
        --num_train_epochs 10 \
        --weight_decay  1e-7 \
        --distill_loss_weight 4.5
```

We can also choose NxM(4x1) as our pruning pattern:
```shell
python ./run_qa_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_bertmini" \
        --do_prune \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000 \
        --cooldown_epochs 5 \
        --learning_rate 4.5e-4 \
        --num_train_epochs 10 \
        --weight_decay  1e-7 \
        --distill_loss_weight 4.5
```

The pruning results of distilbert-base-uncased, bert-base-uncased and bert-large model can be obtained with the following:
```shell
python run_qa_no_trainer.py \
        --model_name_or_path "/path/to/distilbert-base-uncased/dense_finetuned_model" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --do_prune \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_distilbert" \
        --weight_decay 1e-7 \
        --learning_rate 1e-4 \
        --cooldown_epochs 10 \
        --num_train_epochs 20 \
        --distill_loss_weight 3 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000
```
```shell
python run_qa_no_trainer.py \
        --model_name_or_path "/path/to/bert-base-uncased/dense_finetuned_model/" \
        --dataset_name squad \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 12 \
        --do_prune \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_bertbase" \
        --weight_decay 1e-7 \
        --learning_rate 7e-5 \
        --cooldown_epoch 4 \
        --num_train_epochs 10 \
        --distill_loss_weight 4.5 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000
```
```shell
python run_qa_no_trainer.py \
        --model_name_or_path "/path/to/bert-large/dense_finetuned_model/" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 24 \
        --per_device_eval_batch_size 24 \
        --do_prune \
        --num_warmup_steps 1000 \
        --output_dir "./sparse_qa_bertlarge" \
        --weight_decay 0\
        --learning_rate 5e-5 \
        --checkpointing_steps "epoch" \
        --cooldown_epochs 10 \
        --num_train_epochs 40 \
        --distill_loss_weight 3 \
        --target_sparsity 0.8 \
        --pruning_pattern "4x1" \
        --pruning_frequency 1000
```
2:4 sparsity is similar to the above, only the target_sparsity and pruning_pattern need to be changed.

Dense model fine-tune is also supported as following (by setting --do_prune to False):
```shell
python ./run_qa_no_trainer.py \
    --model_name_or_path "prajjwal1/bert-mini" \
    --dataset_name "squad" \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_warmup_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --output_dir "./dense_qa_bertmini"
```

### Results
The snip-momentum pruning method is used by default and the initial dense models are all fine-tuned.
|  Model  | Dataset  |  Sparsity pattern | Element-wise/matmul, Gemm, conv ratio | Dense F1 (mean/max)| Sparse F1 (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----: |:----:| :----: |
| Bert-mini | SQuAD |  4x1  | 0.7993 | 0.7662/0.7687 | 0.7617/0.7627 | -0.78% |
| Bert-mini | SQuAD |  2:4  | 0.4795 | 0.7662/0.7687 | 0.7733/0.7762 | +0.98% |
| Distilbert-base-uncased | SQuAD |  4x1  | 0.7986 | 0.8690 | 0.8615 | -0.86% |
| Distilbert-base-uncased | SQuAD |  2:4  | 0.5000 | 0.8690 | 0.8731/0.8750 | +0.69% |
| Bert-base-uncased | SQuAD |  4x1  | 0.7986 | 0.8859 | 0.8778 | -0.92% |
| Bert-base-uncased | SQuAD |  2:4  | 0.5000 | 0.8859 | 0.8924/0.8940 | +0.91% |
| Bert-large | SQuAD |  4x1  | 0.7988 | 0.9123 | 0.9091 | -0.35% |
| Bert-large | SQuAD |  2:4  | 0.5002 | 0.9123 | 0.9167 | +0.48% |

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)



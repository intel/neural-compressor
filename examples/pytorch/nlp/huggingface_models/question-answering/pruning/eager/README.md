## Examples
we have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.
### [SQuAD](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/question-answering/pruning)

We can train a Squad sparse model with 2:4 pattern:
```shell
python3 ./run_qa_no_trainer.py \
        --model_name_or_path "/path/to/dense_finetuned_model/" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --num_warmup_steps 1000 \
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

We can also choose 4x1 as our pruning pattern:
```shell
python ./run_qa_no_trainer.py \
        --model_name_or_path "/path/to/dense_finetuned_model/" \
        --dataset_name "squad" \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --num_warmup_steps 1000 \
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
    --output_dir "./output_bert-mini"
```

### Results
|  Model  | Dataset  |  Sparsity pattern |Pruning method |Element-wise/matmul, Gemm, conv ratio | Init model | Dense F1 (mean/max)| Sparse F1 (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----: |:----:| :----: | :----: | :----: |
| Bert-Mini  | SQuAD |  4x1  | Snip-momentum |0.7993 | Dense & Finetuned | 0.7662/0.7687 | 0.7617/0.7627 | -0.78% |
| Bert-Mini  | SQuAD |  2:4  | Snip-momentum |0.4795 | Dense & Finetuned | 0.7662/0.7687 | 0.7645/0.7685 | -0.02% |

## References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)


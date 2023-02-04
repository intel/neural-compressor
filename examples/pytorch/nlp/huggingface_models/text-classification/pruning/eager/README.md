Step-by-Step
============
This document describes the step-by-step instructions for reproducing the pruning for Huggingface models.

# Prerequisite
## Environment
```shell
# install dependencies
cd examples/pytorch/nlp/huggingface_models/question-answering/pruning/eager
pip install -r requirements.txt
```

# Pruning
## 1. Train Sparse Model
Train a sparse model with N:M(2:4) pattern on mrpc and sst2:
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-3 \
        --num_train_epochs 15 \
        --weight_decay 1e-3  \
        --do_prune \
        --output_dir "./sparse_mrpc_bertmini" \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --target_sparsity 0.5 \
        --pruning_pattern "2:4" \
        --pruning_frequency 50 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 5
```
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --distill_loss_weight 2.0 \
        --num_train_epochs 15 \
        --weight_decay 5e-5   \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 0 \
        --lr_scheduler_type "constant" \
        --do_prune \
        --output_dir "./sparse_sst2_bertmini" \
        --target_sparsity 0.5 \
        --pruning_pattern "2:4" \
        --pruning_frequency 500
```

NxM (4x1) as pruning pattern on mrpc and sst2:
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertini/dense_finetuned_model" \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-3 \
        --num_train_epochs 15 \
        --weight_decay 1e-3  \
        --do_prune \
        --output_dir "./sparse_mrpc_bertmini" \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --target_sparsity 0.9 \
        --pruning_pattern "4x1" \
        --pruning_frequency 50 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 5
```
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --distill_loss_weight 2.0 \
        --num_train_epochs 15 \
        --weight_decay 5e-5   \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 0 \
        --lr_scheduler_type "constant" \
        --do_prune \
        --output_dir "./sparse_sst2_bertmini" \
        --target_sparsity 0.9 \
        --pruning_pattern "4x1" \
        --pruning_frequency 500
```

Per-channel pruning on mrpc and sst2.
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-3 \
        --num_train_epochs 15 \
        --weight_decay 1e-3  \
        --do_prune \
        --output_dir "./sparse_mrpc_bertmini" \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --target_sparsity 0.9 \
        --pruning_pattern "1xchannel" \
        --pruning_frequency 50 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 5
```
```shell
python3 ./run_glue_no_trainer.py \
        --model_name_or_path "/path/to/bertmini/dense_finetuned_model" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --distill_loss_weight 2.0 \
        --num_train_epochs 15 \
        --weight_decay 5e-5   \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 0 \
        --lr_scheduler_type "constant" \
        --do_prune \
        --output_dir "./sparse_sst2_bertmini" \
        --target_sparsity 0.9 \
        --pruning_pattern "1xchannel" \
        --pruning_frequency 500
```

 Distilbert-base-uncased model pruning on mrpc:
 ```shell
      python run_glue_no_trainer.py \
        --model_name_or_path "path/to/distilbert-base-uncased/dense_finetuned_model" \
        --task_name "mrpc" \
        --max_length 256 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-4\
        --num_train_epochs 120 \
        --weight_decay 0 \
        --cooldown_epochs 40 \
        --sparsity_warm_epochs 0 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 2 \
        --do_prune \
        --output_dir "./sparse_mrpc_distillbert" \
        --target_sparsity 0.9 \
        --pruning_pattern "4x1" \
        --pruning_frequency 50 \
```
2:4 sparsity is similar to the above examples, only the target_sparsity and pruning_pattern need to be changed.

To try to train a sparse model in mixed pattern, the local pruning config can be set as follows:
```python
pruning_configs=[
        {
            "op_names": [".*output", ".*intermediate"], # list of regular expressions, containing the layer names you wish to be included in this pruner.
            "pattern": "1x1",
            "pruning_scope": "local", # the score map is computed corresponding layer's weight.
            "pruning_type": "snip_momentum",
            "sparsity_decay_type": "exp",
            "pruning_op_types": ["Linear"]
        },
        {
            "op_names": [".*query", ".*key", ".*value"],
            "pattern": "4x1",
            "pruning_scope": "global", # the score map is computed out of entire parameters.
            "pruning_type": "snip_momentum",
            "sparsity_decay_type": "exp",
            "max_sparsity_ratio_per_op": 0.98, # Maximum sparsity that can be achieved per layer(iterative pruning).
            "min_sparsity_ratio_per_op": 0.5, # Minimum sparsity that must be achieved per layer(iterative pruning).
            "pruning_op_types": ["Linear"]
        }
]

```

Please be aware that when keywords appear in both the global and the local settings, we select the **local** settings as priority.
```shell
python3 ./run_glue_no_trainer_mixed.py \
        --model_name_or_path "/path/to/dense_finetuned_model/" \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-3 \
        --num_train_epochs 15 \
        --weight_decay 1e-3  \
        --do_prune \
        --output_dir "./sparse_mrpc_bertmini" \
        --cooldown_epochs 5 \
        --sparsity_warm_epochs 1 \
        --target_sparsity 0.9 \
        --lr_scheduler_type "constant" \
        --distill_loss_weight 5
```

We can also train a dense model on glue datasets (by setting --do_prune to False):
```shell
python3 run_glue_no_trainer.py \
        --model_name_or_path "./bert-mini"  \
        --task_name "mrpc" \
        --max_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epoch 5 \
        --weight_decay 5e-5 \
        --output_dir "./dense_mrpc_bertmini"
```
or for sst2:
```shell
python run_glue_no_trainer.py \
         --model_name_or_path "/path/to/dense_pretrained_model/" \
        --task_name "sst2" \
        --max_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        --output_dir "./dense_sst2_bertmini"
```
Results
=======
Please be aware that when the keywords appear in both global and local settings, the **local** settings are given priority.The snip-momentum pruning method is used by default, and the initial dense model is fine-tuned.
### MRPC
|  Model  | Dataset  | Sparsity pattern |Element-wise/matmul, Gemm, conv ratio | Dense Accuracy (mean/max) | Sparse Accuracy (mean/max) | Relative drop |
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| Bert-Mini | MRPC |  4x1  | 0.8804 | 0.8619/0.8752 | 0.8610/0.8722 | -0.34% |
| Bert-Mini | MRPC |  2:4  | 0.4795 | 0.8619/0.8752| 0.8666/0.8689 | -0.72% |
| Bert-Mini | MRPC |  per channel  | 0.66 | 0.8619/0.8752| 0.8629/0.8680 | -0.83% |
| Distilbert-base-uncased | MRPC |  4x1  | 0.8992 | 0.9026 |0.8985 | -0.46% |
| Distilbert-base-uncased | MRPC |  2:4  | 0.5000 | 0.9026 | 0.9088 | +0.69% |

### SST-2
|  Model  | Dataset  |  Sparsity pattern |Element-wise/matmul, Gemm, conv ratio | Dense Accuracy (mean/max) | Sparse Accuracy (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| Bert-Mini | SST-2 |  4x1  | 0.8815 | 0.8660/0.8761 | 0.8651/0.8692 | -0.79% |
| Bert-Mini | SST-2 |  2:4  | 0.4795 | 0.8660/0.8761 | 0.8731/0.8773 | +0.14% |
| Bert-Mini | SST-2 |  per channel  | 0.53 | 0.8660/0.8761 | 0.8651/0.8692 | -0.79% |

# References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)


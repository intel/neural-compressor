Step-by-Step
============

This document presents step-by-step instructions for pruning Huggingface models using the Intel® Neural Compressor.

# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
pip install -r examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/requirements.txt
```

## 2. Prepare Dataset

The dataset will be downloaded automatically from the datasets Hub.
See more about loading [huggingface dataset](https://huggingface.co/docs/datasets/loading_datasets.html)


# Run Examples
Several pruning examples are provided, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.

There are pruning scripts for MPRC and SST2 sparse models (Bert-mini, Distilbert-base-uncased, etc). The sparse model with different patterns ("4x1", "2:4", "1xchannel", etc) can be obtained by modifying "target_sparsity" and "pruning_pattern" parameters. [Pruning Scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/scripts/).

Dense model can also be fine-tuned on glue datasets (by setting --do_prune to False) [Bert-mini MRPC](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/scripts/bertmini_mrpc_dense_finetune.sh)

To try to train a sparse model in mixed pattern [Mixed-patterns Example](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/text-classification/pruning/eager/run_glue_no_trainer_mixed.py), local pruning config can be set as follows:

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
Please be aware that when the keywords appear in both global and local settings, the **local** settings are given priority.

### Results
The snip-momentum pruning method is used by default, and the initial dense model is fine-tuned.
#### MRPC
|  Model  | Dataset  | Sparsity pattern |Element-wise/matmul, Gemm, conv ratio | Dense Accuracy (mean/max) | Sparse Accuracy (mean/max) | Relative drop |
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| Bert-Mini | MRPC |  4x1  | 0.8804 | 0.8619/0.8752 | 0.8610/0.8722 | -0.34% |
| Bert-Mini | MRPC |  2:4  | 0.4795 | 0.8619/0.8752| 0.8666/0.8689 | -0.72% |
| Bert-Mini | MRPC |  per channel  | 0.66 | 0.8619/0.8752| 0.8629/0.8680 | -0.83% |
| Distilbert-base-uncased | MRPC |  4x1  | 0.8992 | 0.9026 |0.8985 | -0.46% |
| Distilbert-base-uncased | MRPC |  2:4  | 0.5000 | 0.9026 | 0.9088 | +0.69% |

#### SST-2
|  Model  | Dataset  |  Sparsity pattern |Element-wise/matmul, Gemm, conv ratio | Dense Accuracy (mean/max) | Sparse Accuracy (mean/max)| Relative drop|
|  :----:  | :----:  | :----: | :----: |:----:|:----:| :----: |
| Bert-Mini | SST-2 |  4x1  | 0.8815 | 0.8660/0.8761 | 0.8651/0.8692 | -0.79% |
| Bert-Mini | SST-2 |  2:4  | 0.4795 | 0.8660/0.8761 | 0.8731/0.8773 | +0.14% |
| Bert-Mini | SST-2 |  per channel  | 0.53 | 0.8660/0.8761 | 0.8651/0.8692 | -0.79% |

# References
* [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)
* [Knowledge Distillation with the Reused Teacher Classifier](https://arxiv.org/abs/2203.14001)






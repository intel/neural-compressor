# "Pruning while distillation" step by step
We have implemented an example of how to prune a model using group lasso during a distillation process.\

# 1 Requirements
nerual_comporessor should be installed successfully.
```
pip install -r requirements.txt
```
# 2 Run the example on **SST2** and external datasets
## 2.1 Modify the configurations
Pay attention to pruning_config.yaml and distillation_config.yaml.\
Please refer to [INC Pruning](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/docs/pruning.md) for usage of pruning_config.yaml\
Since in this example, we do not need a teacher model (teacher model's logits are read from an external dataset), therefore, distillation_config.yaml is not matched with INC's official format.  

## 2.2 Run the code
```
sh run_glue_prune_distillation.sh
```

## Results
| Sparsity ratio | batch size | learning rate | alpha | loss function setting| accuracy |
| :----: |:----:|:----:|:----:|:----:|:----:|
|80|16| 3e-5|2e-3 |100% teacher MSE loss | 88.53% |
|90|16| 2e-5|2e-3 |100% teacher MSE loss | 87.39% |
|80|16| 2e-5|2e-3 |50% teacher MSE loss. 50% student loss | 88.18% |
|80|16| 2e-5|2e-3 |50% teacher KL-Divergence loss, 50% student loss | 88.30% |

If you want to modify the loss function types and their weights, you can refer to distillation configs.

## 2.3 Reference
Original bert-mini distillation baseline can be found in this [notebook](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/examples/notebook/bert_mini_distillation/BERT-Mini-SST2.ipynb)

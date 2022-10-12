# Pytorch Pruner
## Intro
[**Pytorch Pruner**](https://github.com/intel/neural-compressor/tree/master/neural_compressor/experimental/pytorch_pruner) is an INC build-in API which supports a wide range of pruning algorithms, patterns as well as pruning schedulers. Features below are currently supported:
> algorithms: magnitude, snip, snip-momentum\
> patterns: NxM, N:M\
> pruning schedulers: iterative pruning scheduler, oneshot pruning scheduler.

## Usage
### Write a config yaml file
Pytorch pruner is developed based on [pruning](https://github.com/intel/neural-compressor/blob/master/neural_compressor/experimental/pruning.py), therefore most usages are identical. Our API reads in a yaml configuration file to define a Pruning object. Here is an bert-mini example of it:
```yaml
version: 1.0

model:
  name: "bert-mini"
  framework: "pytorch"

pruning:
  approach:
    weight_compression_pytorch:
      # Global settings
      # if start step equals to end step, oneshot pruning scheduler is enabled. Otherwise the API automatically implements iterative pruning scheduler.
      start_step: 0 # step which pruning process begins
      end_step: 0 # step which pruning process ends
      excluded_names: ["classifier", "pooler", ".*embeddings*"] # a global announcement of layers which you do not wish to prune. 
      prune_layer_type: ["Linear"] # the module type which you want to prune (Linear, Conv2d, etc.)
      target_sparsity: 0.9 # the sparsity you want the model to be pruned.
      max_sparsity_ratio_per_layer: 0.98 # the sparsity ratio's maximum which one layer can reach.

      pruners: # below each "Pruner" defines a pruning process for a group of layers. This enables us to apply different pruning methods for different layers in one model.
        # Local settings
        - !Pruner
            extra_excluded_names: [".*query", ".*key", ".*value"] # list of regular expressions, containing the layer names you wish not to be included in this pruner
            pattern: "1x1" # pattern type, we support "NxM" and "N:M"
            update_frequency_on_step: 100 # if use iterative pruning scheduler, this define the pruning frequency.
            prune_domain: "global" # one in ["global", "local"], refers to the score map is computed out of entire parameters or its corresponding layer's weight.
            prune_type: "snip_momentum" # pruning algorithms, refer to pytorch_pruner/pruner.py
            sparsity_decay_type: "exp" # ["linear", "cos", "exp", "cube"] ways to determine the target sparsity during iterative pruning.
        - !Pruner
            extra_excluded_names: [".*output", ".*intermediate"]
            pattern: "4x1"
            update_frequency_on_step: 100
            prune_domain: "global"
            prune_type: "snip_momentum"
            sparsity_decay_type: "exp"
```
Please be awared that when the keywords appear in both global and local settings, we select the **local** settings as priority.
### Coding template:
With a settled config file, we provide a template for implementing pytorch_pruner API:
```python
model = Model()
criterion = Criterion()
optimizer = Optimizer()
args = Args()

from neural_compressor.experimental.pytorch_pruner.pruning import Pruning

pruner = Pruning("path/to/your/config.yaml")
if args.do_prune:
    pruner.update_items_for_all_pruners(start_step=int(args.sparsity_warm_epochs * num_iterations), end_step=int(total_iterations))  ##iterative
else:
   pruner.update_items_for_all_pruners(start_step=total_iterations+1, end_step=total_iterations+1) ## remove the pruner
pruner.model = model
pruner.on_train_begin()
for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        pruner.on_step_begin(step)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        pruner.on_before_optimizer_step()
        optimizer.step()
        pruner.on_after_optimizer_step()
        optimizer.zero_grad()
    
    model.eval()
    for step, batch in enumerate(val_dataloader):
        ...
```
For more usage, please refer to our example codes below.

## Examples
we have provided several pruning examples, which are trained on different datasets/tasks, use different sparsity patterns, etc. We are working on sharing our sparse models on HuggingFace.
### [SQuAD](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/question-answering/pruning)
We can train a sparse model with NxM (2:4) pattern:
```
python3 ./run_qa_no_trainer.py \
            --model_name_or_path "/path/to/dense_finetuned_model/" \
            --pruning_config "./bert_mini_2in4.yaml" \
            --dataset_name "squad" \
            --max_seq_length "384" \
            --doc_stride "128" \
            --per_device_train_batch_size "8" \
            --weight_decay "1e-7" \
            --learning_rate "1e-4" \
            --num_train_epochs "10" \
            --teacher_model_name_or_path "/path/to/dense_finetuned_model/" \
            --distill_loss_weight "8.0"
```
We can also choose 4x1 as our pruning pattern:
```
python ./run_qa_no_trainer.py \
        --model_name_or_path "/path/to/dense_finetuned_model/" \
        --pruning_config "./bert_mini_4x1.yaml" \
        --dataset_name "squad" \
        --max_seq_length "384" \
        --doc_stride "128" \
        --per_device_train_batch_size "16" \
        --per_device_eval_batch_size "16" \
        --num_warmup_steps "1000" \
        --do_prune \
        --cooldown_epochs "5" \
        --learning_rate "4.5e-4" \
        --num_train_epochs "10" \
        --weight_decay  "1e-7" \
        --output_dir "pruned_squad_bert-mini" \
        --teacher_model_name_or_path "/path/to/dense_finetuned_model/" \
        --distill_loss_weight "4.5"
```
Dense model training is also supported as following (by setting --do_prune to False):
```
python \
    ./run_qa_no_trainer.py \
    --model_name_or_path "prajjwal1/bert-mini" \
    --pruning_config "./bert_mini_4x1.yaml" \
    --dataset_name "squad" \
    --max_seq_length "384" \
    --doc_stride "128" \
    --per_device_train_batch_size "8" \
    --per_device_eval_batch_size "16" \
    --num_warmup_steps "1000" \
    --learning_rate "5e-5" \
    --num_train_epochs "5" \
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

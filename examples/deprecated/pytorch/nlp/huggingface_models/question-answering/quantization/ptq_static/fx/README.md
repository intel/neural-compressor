Step-by-Step
============

This document is used to list the steps of reproducing quantization and benchmarking results.

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are all in requirements, please install as following.
```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
python -u ./run_qa.py \
        --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
        --dataset_name squad \
        --do_eval \
        --do_train \
        --max_seq_length 384 \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir saved_results \
        --tune \
        --overwrite_output_dir \
        --dataloader_drop_last
```
> NOTE
>
> `saved_results` is the path to finetuned output_dir
or
```bash
sh run_quant.sh --topology=topology_name --input_model=model_name_or_path
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --int8=true --config=saved_results
# fp32
sh run_benchmark.sh --topology=topology_name --mode=performance
```
## 3. Validated Model List
<table>
<thead>
  <tr>
    <th>Topology Name</th>
    <th>Model Name</th>
    <th>Dataset/Task Name</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>bert_large_SQuAD</td>
    <td><a href="https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad">bert-large-uncased-whole-word-masking-finetuned-squad</a></td>
    <td><a href="https://huggingface.co/datasets/squad">squad</a></td>
  </tr>
</tbody>
</table>

# Tutorial of Enabling NLP Models with Intel® Neural Compressor.
## 1. Intel® Neural Compressor supports two usages:
1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.
## 2. Code Prepare

We need to update `run_qa.py` like below:

```python
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
)

eval_dataloader = trainer.get_eval_dataloader()
batch_size = eval_dataloader.batch_size
metric_name = "eval_f1"

def take_eval_steps(model, trainer, metric_name, save_metrics=False):
    trainer.model = model
    metrics = trainer.evaluate()
    return metrics.get(metric_name)

def eval_func(model):
    return take_eval_steps(model, trainer, metric_name)

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=eval_dataloader,
                           eval_func=eval_func)
q_model.save(training_args.output_dir)
```

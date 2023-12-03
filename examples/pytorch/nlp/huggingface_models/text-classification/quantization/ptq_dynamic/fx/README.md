Step-by-Step
============

This document is used to introduce the details about how to quantize the model with `PostTrainingDynamic` on the text classification task and obtain the benchmarking results. 

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in requirements, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx
pip install -r requirements.txt
```

# Run
## 1. Quantization
```shell
python run_glue.py \
        --model_name_or_path yoshitomo-matsubara/bert-large-uncased-rte \
        --task_name rte \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size 16 \
        --no_cuda \
        --output_dir saved_results\
        --tune \
        --overwrite_output_dir
```
> NOTE
>
> `saved_results` is the path to finetuned output_dir.

or
```bash
sh run_quant.sh --topology=topology_name --input_model=model_name_or_path
```
## 2. Benchmark
```bash
# int8
sh run_benchmark.sh --topology=topology_name --mode=performance --int8=true --input_model=saved_results
# fp32
sh run_benchmark.sh --topology=topology_name --mode=performance --input_model=model_name_or_path
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
    <td>bert_large_RTE_dynamic</td>
    <td><a href="https://huggingface.co/yoshitomo-matsubara/bert-large-uncased-rte">yoshitomo-matsubara/bert-large-uncased-rte</a></td>
    <td>rte</td>
  </tr>
  <tr>
    <td>xlm-roberta-base_MRPC_dynamic</td>
    <td><a href="https://huggingface.co/Intel/xlm-roberta-base-mrpc">Intel/xlm-roberta-base-mrpc</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>distilbert_base_MRPC_dynamic</td>
    <td><a href="https://huggingface.co/textattack/distilbert-base-uncased-MRPC">textattack/distilbert-base-uncased-MRPC</a></td>
    <td>mrpc</td>
  </tr>
  <tr>
    <td>albert_base_MRPC</td>
    <td><a href="https://huggingface.co/textattack/albert-base-v2-MRPC">textattack/albert-base-v2-MRPC</a></td>
    <td>mrpc</td>
  </tr>
</tbody>
</table>

# HuggingFace Model Hub
## 1. To upstream into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and int8 model weights in the format of [transformers](https://github.com/huggingface/transformers). 
```py
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into model hub and reuse them with our `OptimizedModel` API.

## 2. To download from HuggingFace model hub
We provide an API `OptimizedModel` to initialize INT8 models from HuggingFace model hub and its usage is same as the model class provided by [transformers](https://github.com/huggingface/transformers).
```python
from neural_compressor.utils.load_huggingface import OptimizedModel
model = OptimizedModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
```
We also upstreamed several INT8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.
----

# Tutorial of Enabling NLP Models with Intel® Neural Compressor
## 1. Intel® Neural Compressor supports two usages:
1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

## 2. Code Prepare
We update `run_glue.py` like belows:

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
tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(approach="dynamic", tuning_criterion=tuning_criterion)
q_model = fit(model, conf=conf, eval_func=eval_func)
q_model.save(training_args.output_dir)
```

Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/text-classification/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install BERT model

```bash
pip install transformers
```

#### Install dependency

```shell
pip install -r requirements.txt
```

#### Install PyTorch
```shell
pip install torch
```

## 2. Prepare pretrained model

Before use Intel® Neural Compressor, you should fine tune the model to get pretrained model or reuse fine-tuned models in [model hub](https://huggingface.co/models), You should also install the additional packages required by the examples.

# Start to neural_compressor tune for Model Quantization
 - Here we implemented several models in fx mode.
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx
```
## Glue task

### 1. To get the tuned model and its accuracy: 
```bash
python -u ./run_glue.py \
        --model_name_or_path bert-large-rte \
        --task_name rte \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size 16 \
        --no_cuda \
        --output_dir ./int8_model_dir \
        --tune \
        --overwrite_output_dir
``` 

### 2. To get the benchmark of tuned model, includes Batch_size and Throughput: 

```bash
python -u ./run_glue.py \
        --model_name_or_path ./int8_model_dir \
        --task_name sst2 \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size 1 \
        --no_cuda \
        --output_dir ./output_log \
        --benchmark \
        --int8 \
        --overwrite_output_dir
```

# HuggingFace model hub
## To upstream into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and int8 model weights in the format of [transformers](https://github.com/huggingface/transformers). 
```
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
...

save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into model hub and reuse them with our `OptimizedModel` API.

## To download into HuggingFace model hub
We provide an API `OptimizedModel` to initialize int8 models from HuggingFace model hub and its usage is the same as the model class provided by [transformers](https://github.com/huggingface/transformers).
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

We also upstreamed several int8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.

----
----
## This is a tutorial of how to enable NLP model with Intel® Neural Compressor.


### Intel® Neural Compressor supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As MRPC's metrics are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Write Yaml config file

In examples directory, there is conf.yaml. We could remove most of the items and only keep mandatory item for tuning.

```yaml
model:
  name: bert
  framework: pytorch_fx

device: cpu

quantization:
  approach: post_training_dynamic_quant

tuning:
  accuracy_criterion:
    relative: 0.01
  exit_policy:
    timeout: 0
    max_trials: 300
  random_seed: 9527
```

Here we set accuracy target as tolerating 0.01 relative accuracy loss of baseline. The default tuning strategy is basic strategy. The timeout 0 means early stop as well as a tuning config meet accuracy target.

> **Note** : neural_compressor does NOT support "mse" tuning strategy for pytorch framework

### Code Prepare

We just need update run_squad_tune.py and run_glue.py like below

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

eval_dataloader = trainer.get_eval_dataloader()
batch_size = eval_dataloader.batch_size

metric_name = "eval_f1"
def take_eval_steps(model, trainer, metric_name, save_metrics=False):
    trainer.model = model
    metrics = trainer.evaluate()
    if save_metrics:
        trainer.save_metrics("eval", metrics)
    logger.info("{}: {}".format(metric_name, metrics.get(metric_name)))
    logger.info("Throughput: {} samples/sec".format(metrics.get("eval_samples_per_second")))
    return metrics.get(metric_name)
def eval_func(model):
    return take_eval_steps(model, trainer, metric_name)

from neural_compressor.experimental import Quantization, common
if (
    not training_args.dataloader_drop_last
    and eval_dataset.shape[0] % training_args.per_device_eval_batch_size != 0
):
    raise ValueError(
        "The number of samples of the dataset is not a multiple of the batch size."
        "Use --dataloader_drop_last to overcome."
    )
calib_dataloader = eval_dataloader
quantizer = Quantization('conf.yaml')
quantizer.eval_func = eval_func
quantizer.calib_dataloader = calib_dataloader
quantizer.model = common.Model(model)
model = quantizer.fit()
model.save(training_args.output_dir)
```

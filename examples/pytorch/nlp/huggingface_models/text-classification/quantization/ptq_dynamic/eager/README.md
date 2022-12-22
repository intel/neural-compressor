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

#### Install transformers

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
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/eager
```
## Glue task
### 1. To get the tuned model and its accuracy:

```bash
sh run_tuning.sh --topology=topology_name --dataset_location=/path/to/glue/data/dir --input_model=/path/to/checkpoint/dir
```
> NOTE
>
> topology_name can be:{"bert_base_MRPC", "distilbert_base_MRPC", "albert_base_MRPC", "funnel_MRPC", "bart_WNLI", "mbart_WNLI", "xlm_roberta_MRPC", "gpt2_MRPC", "xlnet_base_MRPC", "transfo_xl_MRPC", "ctrl_MRPC", "xlm_MRPC"}
>
> /path/to/checkpoint/dir is the path to finetune output_dir 

or

```bash
python -u ./run_glue_tune.py \
        --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size 16 \
        --no_cuda \
        --output_dir ./int8_model_dir \
        --tune \
        --overwrite_output_dir
``` 

### 2. To get the benchmark of tuned model, includes batch_size and throughput: 

```bash
python -u ./run_glue_tune.py \
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

----

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

Examples of enabling Intel® Neural Compressor
============================================================

This is a tutorial of how to enable BERT model with Intel® Neural Compressor.

## User Code Analysis

Intel® Neural Compressor supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics in tuning.metrics field of model-specific yaml config file.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As BERT's matricses are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Write Yaml config file

In examples directory, there is conf.yaml. We could remove most of the items and only keep mandatory item for tuning.

```yaml
model:
  name: bert
  framework: pytorch

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

We just need update run_squad_tune.py and run_glue_tune.py like below

```python
if model_args.tune:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    def eval_func_for_nc(model_tuned):
        trainer.model = model_tuned
        result = trainer.evaluate(eval_dataset=eval_dataset)
        bert_task_acc_keys = ['eval_f1', 'eval_accuracy', 'mcc', 'spearmanr', 'acc']
        for key in bert_task_acc_keys:
            if key in result.keys():
                logger.info("Finally Eval {}:{}".format(key, result[key]))
                acc = result[key]
                break
        return acc
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    calib_dataloader = trainer.get_train_dataloader()
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func_for_nc
    q_model = quantizer()
    q_model.save(training_args.output_dir)
    exit(0)
```

### Using Shapley MSE as Objective

Shapley values originate from cooperative game theory that come with desirable properties, and now are widely used as a tool to fulfill Explainable AI. The run_glue_tune_with_shap.py is designed to help build a bert-based model using Shapley MSE as an objective. Here, the Shapley MSE means that we can get one result from FP32 and several results from INT8 model, so we use MSE to calculate how different between the two shapley values. It can reflect the explainability of INT8 model. 
> **Note** : run_glue_tune_with_shap.py is the example of "SST2" task. If you want to execute other glue task, you may take some slight change under "ShapleyMSE" class.  





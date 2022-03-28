Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/question-answering/README.md).

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install BERT model

```bash
pip install transformers==4.10.0
```

#### Install dependency

```shell
pip install -r requirements.txt
```

#### Install PyTorch
```shell
pip install torch
```

# Start to neural_compressor tune for Model Quantization

 - Here we implemented several models in fx mode.

```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx

python -u ./run_qa.py \
        --model_name_or_path "bert-large-uncased-whole-word-masking-finetuned-squad" \
        --dataset_name "squad" \
        --do_eval \
        --do_train \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir /path/to/checkpoint/dir \
        --tune \
        --overwrite_output_dir \
        --dataloader_drop_last
```
> NOTE
>
> /path/to/checkpoint/dir is the path to finetune output_dir


---------------------

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
  approach: post_training_static_quant
  op_wise: {
            # PyTorch limitation: PyTorch unsupport specific qconfig for function when version <=1.10, will remove furture.
            'default_qconfig': {
              'activation':  {'dtype': ['fp32']},
              'weight': {'dtype': ['fp32']}
            },
           }

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

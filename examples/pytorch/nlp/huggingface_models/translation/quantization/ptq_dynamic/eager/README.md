Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/seq2seq/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install transformers

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
```shell
cd examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/eager
sh run_tuning.sh --topology=topology_name
```
> NOTE
>
> topology_name can be:{"t5_WMT_en_ro", "marianmt_WMT_en_ro"}
>

Examples of enabling Intel® Neural Compressor
============================================================

This is a tutorial of how to enable BERT model with Intel® Neural Compressor.

# User Code Analysis

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

For translation task,We need update run_translation.py like below

```python
if model_args.tune:
    def eval_func_for_nc(model):
        trainer.model = model
        results = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        task_metrics_keys = ['eval_bleu','eval_rouge1','eval_rouge2','eval_rougeL','eval_rougeLsum']
        for key in task_metrics_keys:
            if key in results.keys():
                logger.info("Finally Eval {}:{}".format(key, results[key]))
                if 'bleu' in key:
                    acc = results[key]
                    break
                if 'rouge' in key:
                    acc = sum([v for k,v in results.items() if "rouge" in k])/4
                    break
        return acc
    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    calib_dataloader = trainer.get_train_dataloader()
    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func_for_nc
    q_model = quantizer.fit()
    q_model.save(training_args.output_dir)
    exit(0)
```



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

## 2. Prepare pretrained model

Before use Intel® Neural Compressor, you should fine tune the model to get pretrained model, You should also install the additional packages required by the examples:

### Seq2seq

#### summarization_billsum task

##### Prepare dataset
```shell
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum.tar.gz
tar -xzvf billsum.tar.gz
```
##### Finetune command
```shell
export TASK_NAME=summarization_billsum

python run_seq2seq_tune.py \
  --model_name_or_path google/pegasus-billsum \
  --do_train \
  --do_eval \
  --task $TASK_NAME \
  --data_dir /path/to/data/dir \
  --output_dir /path/to/checkpoint/dir \
  --overwrite_output_dir \
  --predict_with_generate \
  --max_source_length 1024 \
  --max_target_length=256 \
  --val_max_target_length=256 \
  --test_max_target_length=256 
```
for example /path/to/data/dir can be `./billsum`

#### translation_en_ro task

##### Finetune command
```shell
export TASK_NAME=translation

python run_seq2seq_tune.py \
  --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --task $TASK_NAME \
  --data_dir /path/to/data/dir \
  --output_dir /path/to/checkpoint/dir \
  --overwrite_output_dir \
  --predict_with_generate 
```
for example /path/to/data/dir can be `../test_data/wmt_en_ro`

> NOTE 
>
> model_name_or_path : Path to pretrained model or model identifier from huggingface.co/models
>
> task : Task name, summarization (or summarization_{dataset} for pegasus) or translation

Where task name can be one of summarization_{summarization dataset name},translation_{language}\_to\_{language}.

Where summarization dataset can be one of xsum,billsum etc.

Where output_dir is path of checkpoint which be created by fine tuning.

* After fine tuning, you can get a checkpoint dir which include pretrained model, tokenizer and training arguments. This checkpoint dir will be used by neural_compressor tuning as below.


# Start to neural_compressor tune for Model Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/seq2seq/quantization/ptq_dynamic/eager
```

## Seq2seq task

```bash
sh run_tuning.sh --topology=topology_name --dataset_location=/path/to/seq2seq/data/dir --input_model=/path/to/checkpoint/dir
```
> NOTE
>
> topology_name can be:{"t5_WMT_en_ro", "marianmt_WMT_en_ro", "pegasus_billsum"}
>
> /path/to/checkpoint/dir is the path to output_dir set in finetune. 
>
> /path/to/seq2seq/data/dir is the path to data_dir set in finetune.
>
> for example,
>
> `examples/test_data/wmt_en_ro` for translation task
>
> `examples/seq2seq/billsum` for summarization task

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

For seq2seq task,We need update run_seq2seq_tune.py like below

```python
if training_args.tune:
    def eval_func_for_nc(model):
        trainer.model = model
        results = trainer.evaluate(
            eval_dataset=eval_dataset,metric_key_prefix="val", max_length=data_args.val_max_target_length, num_beams=data_args.eval_beams
        )
        assert data_args.task.startswith("summarization") or data_args.task.startswith("translation") , \
            "data_args.task should startswith summarization or translation"
        task_metrics_keys = ['val_bleu','val_rouge1','val_rouge2','val_rougeL','val_rougeLsum']
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
    q_model = quantizer()
    q_model.save(training_args.output_dir)
    exit(0)
```



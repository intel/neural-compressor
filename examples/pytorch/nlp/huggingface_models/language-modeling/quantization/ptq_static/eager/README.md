Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](../../../../common/README.md) and [README](../../../../common/examples/language-modeling/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install transformers

```bash
cd examples/pytorch/nlp/huggingface_models/common
python setup.py install
```

> **Note**
>
> Please don't install public transformers package.

#### Install dependency

```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/eager
pip install -r requirements.txt
```

#### Install PyTorch
```shell
pip install torch>=1.7
```

## 2. Prepare pretrained model

Before use Intel® Neural Compressor, you should fine tune the model to get pretrained model, You should also install the additional packages required by the examples:

#### Language-modeling
##### Finetune command
```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/eager

python run_clm_tune.py \
  --model_name_or_path microsoft/DialoGPT-small \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --output_dir /path/to/checkpoint/dir
```
> NOTE 
>
> the metric we use is eval loss, not perplexity, because the perplexity is too sensitive. Until now, we have enabled dialogpt, reformer, ctrl models. The accuracy of the int8 model is relative less than 0.01 to fp32, except for ctrl, which is relative less than 0.05.
>
> model_name_or_path : Path to pretrained model or model identifier from huggingface.co/models
>
> dataset_name : Dataset name, can be one of {wikitext, crime_and_punish}.
>
> dataset_config_name : just for dialogpt: wikitext-2-raw-v1.

# Start to neural_compressor tune for Model Quantization
```shell
cd examples/pytorch/nlp/huggingface_models/quantization/ptq_dynamic/eager
```
## Glue task

```bash
sh run_tuning.sh --topology=topology_name --dataset_location=/path/to/glue/data/dir --input_model=/path/to/checkpoint/dir
```
> NOTE
>
> topology_name can be:{"bert_base_MRPC", "distilbert_base_MRPC", "albert_base_MRPC", "funnel_MRPC", "bart_WNLI", "mbart_WNLI", "xlm_roberta_MRPC", "gpt2_MRPC", "xlnet_base_MRPC", "transfo_xl_MRPC", "ctrl_MRPC", "xlm_MRPC"}
>
> /path/to/checkpoint/dir is the path to finetune output_dir 

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

## Language-modeling task
```bash
sh run_tuning.sh --topology=topology_name --input_model=/path/to/checkpoint/dir
```
> NOTE
>
> topology_name can be one of {dialogpt_wikitext, reformer_crime_and_punishment}
>
> /path/to/checkpoint/dir is the path to finetune output_dir 

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
  approach: post_training_static_quant

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

For language modeling task,We need update run_clm_tune.py like below

```python
if training_args.tune:
    def eval_func_for_nc(model_tuned):
        trainer.model = model_tuned
        eval_output = trainer.evaluate(eval_dataset=eval_dataset)
        perplexity = math.exp(eval_output["eval_loss"])
        results = {"perplexity":perplexity,"eval_loss":eval_output["eval_loss"],\
                    "eval_samples_per_second":eval_output['eval_samples_per_second']}
        clm_task_metrics_keys = ["perplexity"]
        for key in clm_task_metrics_keys:
            if key in results.keys():
                logger.info("Finally Eval {}:{}".format(key, results[key]))
                if key=="perplexity":
                    perplexity = results[key]
                    break
        return 100-perplexity

    from neural_compressor.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = trainer.get_eval_dataloader()
    quantizer.eval_func = eval_func_for_nc
    q_model = quantizer.fit()
    q_model.save(training_args.tuned_checkpoint)
    exit(0)
```


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
cd examples/pytorch/nlp/huggingface_models/common
python setup.py install
```

> **Note**
>
> Please don't install public transformers package.

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

### XLNet
For glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...)

```shell
cd examples/pytorch/nlp/huggingface_models/common/examples/text-classification
export TASK_NAME=MRPC

python run_glue.py 
  --model_name_or_path xlnet-base-cased 
  --task_name $TASK_NAME  
  --do_train 
  --do_eval 
  --max_seq_length 256 
  --per_device_train_batch_size 32 
  --learning_rate 5e-5 
  --num_train_epochs 5 
  --output_dir /path/to/checkpoint/dir
```

# Start to neural_compressor tune for Model Quantization

 - We recommand you to try [Huggingface/Optimum](https://github.com/huggingface/optimum) for static quantization, which intergrated our tool[INC] for quantization. Fx mode in Optimum is better for static quantization, which can automatically insert quant/dequant modules.

 - Here we implemented several models in eager mode that fx mode is not ready for. We already inserted quant/dequant modules in our local [Transformers](examples/pytorch/nlp/huggingface_models/common).

```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/eager/
```
## Glue task

```bash
sh run_tuning.sh --topology=topology_name --input_model=/path/to/checkpoint/dir [--dataset_location=/path/to/glue/data/dir]
```
> NOTE
>
> topology_name should be one of them: {"xlm-roberta-base_MRPC", "flaubert_MRPC", "barthez_MRPC", "longformer_MRPC", "layoutlm_MRPC", "deberta_MRPC", "squeezebert_MRPC", "xlnet_base_cased_MRPC", "roberta_base_MRPC", "camembert_base_MRPC"}
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
  name: xlnet
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
if training_args.tune:
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
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = trainer.get_eval_dataloader()
    quantizer.eval_func = eval_func_for_nc
    q_model = quantizer()
    q_model.save(training_args.tuned_checkpoint)
    exit(0)
```

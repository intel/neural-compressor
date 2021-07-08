Step-by-Step
============

This document is used to list steps of reproducing PyTorch BERT tuning zoo result.
Original BERT documents please refer to [BERT README](BERT_README.md) and [README](examples/text-classification/README.md).

> **Note**
>
> Dynamic Quantization is the recommended method for huggingface models. 

# Prerequisite

## 1. Installation

### Python Version

Recommend python 3.6 or higher version.

#### Install BERT model

```bash
cd examples/pytorch/eager/huggingface_models
python setup.py install
```

> **Note**
>
> Please don't install public transformers package.

#### Install dependency

```shell
cd examples/text-classification
pip install -r requirements.txt

cd ../seq2seq
pip install -r requirements.txt
```

#### Install PyTorch
```shell
# Install Dependencies
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
# Install pytorch from source
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git reset --hard 24aac321718d58791c4e6b7cfa50788a124dae23
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
pip install torchvision==0.7.0 --no-deps
```

## 2. Prepare pretrained model

Before use Intel® Low Precision Optimization Tool, you should fine tune the model to get pretrained model, You should also install the additional packages required by the examples:

### Text-classification

#### BERT
* For BERT base and glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...)

```shell
export TASK_NAME=MRPC

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```
> NOTE 
>
> model_name_or_path : Path to pretrained model or model identifier from huggingface.co/models
>
> task_name : where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

please refer to [BERT base scripts and instructions](examples/text-classification/README.md#PyTorch version).

* After fine tuning, you can get a checkpoint dir which include pretrained model, tokenizer and training arguments. This checkpoint dir will be used by lpot tuning as below.

#### XLM-RoBERTa
For BERT base and glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...)

```shell
cd examples/text-classification
export TASK_NAME=MRPC

python run_glue.py 
  --model_name_or_path xlm-roberta-base 
  --task_name $TASK_NAME 
  --do_train 
  --do_eval 
  --max_seq_length 128 
  --per_device_train_batch_size 32 
  --learning_rate 2e-5 
  --num_train_epochs 2 
  --output_dir /path/to/checkpoint/dir
```

#### XLNet
For BERT base and glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...)

```shell
cd examples/text-classification
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

#### GPT2
For BERT base and glue tasks(task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI...). First you need to create a model with appropriate number of labels. For example, MRPC is a two-way sentence pair classification task.

```python
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Save the model and tokenizer in a directory
model.save_pretrained("directory")
tokenizer.save_pretrained("directory")
```

fine tuning command like this.
```shell
cd examples/text-classification
export MODEL_PATH=directory
export TASK_NAME=MRPC

python run_glue.py 
  --model_name_or_path $MODEL_PATH
  --task_name $TASK_NAME 
  --do_train 
  --do_eval 
  --max_seq_length 128 
  --per_device_train_batch_size 1 
  --learning_rate 2e-5 
  --num_train_epochs 1 
  --output_dir /path/to/checkpoint/dir
  
```



### Seq2seq

#### Install dependency
```shell
cd examples/pytorch/eager/huggingface_models/examples/seq2seq
pip install -r requirements.txt
```

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

* After fine tuning, you can get a checkpoint dir which include pretrained model, tokenizer and training arguments. This checkpoint dir will be used by lpot tuning as below.

#### Language-modeling
##### Finetune command
```shell
cd examples/pytorch/eager/huggingface_models/examples/language-modeling
```
```shell
python run_clm.py \
  --model_name_or_path microsoft/DialoGPT-small \
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --do_eval \
  --output_dir /path/to/checkpoint/dir
```
> NOTE 
>
> Until now, enabled dialogpt, reformer.
>
> model_name_or_path : Path to pretrained model or model identifier from huggingface.co/models
>
> dataset_name : Dataset name, can be one of {wikitext, crime_and_punish}.
>
> dataset_config_name : just for dialogpt: wikitext-2-raw-v1.

# Start to lpot tune for Model Quantization
```shell
cd examples/pytorch/eager/huggingface_models
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


# Start to lpot tune for Model Pruning

Below are example NLP tasks for model pruning together with task specific fine-tuning.
It requires the pre-trained bert-base sparsity model `Intel/bert-base-uncased-sparse-70-unstructured` from Intel Huggingface portal.
The pruning configuration is specified in yaml file i.e. prune.yaml.

## MNLI task

```bash
python examples/text-classification/run_glue_no_trainer_prune.py --task_name mnli --max_length 128
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured
       --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```

## SST-2 task

```bash
python examples/text-classification/run_glue_no_trainer_prune.py --task_name sst2 --max_length 128
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured
       --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```

## QQP task

```bash
python examples/text-classification/run_glue_no_trainer_prune.py --task_name qqp --max_length 128
       --model_name_or_path Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier
       --seed 42 --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3
       --weight_decay 0.1 --num_warmup_steps 1700 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt
```

## QNLI task

```bash
python examples/text-classification/run_glue_no_trainer_prune.py --task_name qnli --max_length 128
       --model_name_or_path Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier
       --seed 42 --per_device_train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3
       --weight_decay 0.1 --num_warmup_steps 500 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt
```

## QnA task

```bash
python examples/question-answering/run_qa_no_trainer_prune.py
       --model_name_or_path Intel/bert-base-uncased-sparse-70-unstructured --dataset_name squad
       --max_seq_length 384 --doc_stride 128 --weight_decay 0.01 --num_warmup_steps 900 --learning_rate 1e-4
       --num_train_epochs 2 --per_device_train_batch_size 12 --output_dir /path/to/output_dir
       --prune --config prune.yaml --output_model /path/to/output/model.pt --seed 5143
```


Examples of enabling Intel® Low Precision Optimization Tool
============================================================

This is a tutorial of how to enable BERT model with Intel® Low Precision Optimization Tool.

# User Code Analysis

Intel® Low Precision Optimization Tool supports two usages:

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

> **Note** : lpot does NOT support "mse" tuning strategy for pytorch framework

### Code Prepare

We just need update run_squad_tune.py and run_glue_tune.py like below

```python
if training_args.tune:
    def eval_func_for_lpot(model_tuned):
        trainer = Trainer(
            model=model_tuned,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        result = trainer.evaluate(eval_dataset=eval_dataset)
        bert_task_acc_keys = ['eval_f1', 'eval_accuracy', 'mcc', 'spearmanr', 'acc']
        for key in bert_task_acc_keys:
            if key in result.keys():
                logger.info("Finally Eval {}:{}".format(key, result[key]))
                acc = result[key]
                break
        return acc
    from lpot.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    calibration_dataset = quantizer.dataset('bert', dataset=eval_dataset,
                                         task="classifier", model_type=config.model_type)
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = common.DataLoader(
        calibration_dataset, batch_size=training_args.per_device_eval_batch_size)
    quantizer.eval_func = eval_func_for_lpot
    q_model = quantizer()
    q_model.save(training_args.tuned_checkpoint)
    exit(0)
```

For seq2seq task,We need update run_seq2seq_tune.py like below

```python
if training_args.tune:
    def eval_func_for_lpot(model):
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
    from lpot.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = common.DataLoader(
                                            eval_dataset, 
                                            batch_size=training_args.eval_batch_size,
                                            collate_fn=Seq2SeqDataCollator_lpot(tokenizer, data_args, training_args.tpu_num_cores)
                                            )
    quantizer.eval_func = eval_func_for_lpot
    q_model = quantizer()
    q_model.save(training_args.tuned_checkpoint)
    exit(0)
```

For language modeling task,We need update run_clm_tune.py like below

```python
if training_args.tune:
    def eval_func_for_lpot(model_tuned):
        trainer = Trainer(
            model=model_tuned,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
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
    from lpot.experimental import Quantization, common
    quantizer = Quantization("./conf.yaml")
    quantizer.model = common.Model(model)
    quantizer.calib_dataloader = common.DataLoader(
                                            eval_dataset, 
                                            batch_size=training_args.eval_batch_size,
                                            collate_fn=default_data_collator_lpot
                                            )
    quantizer.eval_func = eval_func_for_lpot
    q_model = quantizer()
    q_model.save(training_args.tuned_checkpoint)
    exit(0)
```


Step-by-Step
============

This document is used to list steps of reproducing Huggingface models tuning zoo result.

# Prerequisite

## Environment
Recommend python 3.6 or higher version.
```shell
cd examples/pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/fx
pip install transformers==4.10.0
pip install -r requirements.txt
pip install torch
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment). 

# Quantization
```shell
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

# Tutorial of How to Enable NLP Model with Intel® Neural Compressor.
### Intel® Neural Compressor supports two usages:

1. User specifies fp32 'model', calibration dataset 'q_dataloader', evaluation dataset "eval_dataloader" and metrics.
2. User specifies fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func" which encapsulates the evaluation dataset and metrics by itself.

As MRPC's metrics are 'f1', 'acc_and_f1', mcc', 'spearmanr', 'acc', so customer should provide evaluation function 'eval_func', it's suitable for the second use case.

### Code Prepare

We just need update run_qa.py like below

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

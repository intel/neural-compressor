Step-by-Step
============

This document introduces steps of reproducing Intel Optimized PyTorch bert-base-cased/uncased models tuning results via Neural Compressor with quantization aware training.

Our example comes from [Huggingface/transformers](https://github.com/huggingface/transformers)


# Prerequisite

## Environment

PyTorch >=1.12.0 is recommended for pytorch_fx backend and huggingface/transformers.

  ```shell
  cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/qat/fx
  pip install -r requirements.txt
  ```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

## Prepare fine-tuned model

  ```shell
  python run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir bert_model
  ```

# Run

## Enable bert-base-cased/uncased example with the auto quantization aware training strategy of Neural Compressor.

  The changes made are shown as follows:
  * edit run_glue.py:  
    - For quantization, We used neural_compressor in it.  
    - For training, we enbaled early stop strategy.  

## Get the tuned model and its accuracy: 

    bash run_tuning.sh --input_model=./bert_model  --output_model=./saved_results

or

    python run_glue.py \
        --model_name_or_path ${input_model} \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${output_model} --overwrite_output_dir \
        --eval_steps 300 \
        --save_steps 300 \
        --greater_is_better True \
        --load_best_model_at_end True \
        --evaluation_strategy steps \
        --save_strategy steps \
        --metric_for_best_model f1 \
        --save_total_limit 1 \
        --tune

## Get the benchmark of tuned model, including Batch_size and Throughput: 

    bash run_benchmark.sh --input_model=./bert_model --config=./saved_results --mode=benchmark --int8=true/false

or

    python run_glue.py \
        --model_name_or_path ${input_model}/${tuned_checkpoint} \
        --task_name ${task_name} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_eval_batch_size ${batch_size} \
        --per_device_train_batch_size ${batch_size} \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --metric_for_best_model f1 \
        --output_dir ./output_log --overwrite_output_dir \
        --benchmark [--int8]


# HuggingFace model hub
## Upstream model files into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and int8 model weights in the format of [transformers](https://github.com/huggingface/transformers). 
```
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
...

save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into model hub and reuse them with our `OptimizedModel` API.

## Download the model into HuggingFace model hub
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

We have upstreamed several int8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.

----
----
## This is a tutorial about how to enable NLP model with Intel® Neural Compressor.


### Intel® Neural Compressor supports usage:
* User needs to specify fp32 'model', calibration dataset 'q_dataloader' and a custom "eval_func", which encapsulates the evaluation dataset and metrics by itself.

### Code Prepare

The updated run_glue.py is shown as below

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

from neural_compressor.training import prepare_compression
from neural_compressor.config import QuantizationAwareTrainingConfig
conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model, conf)
compression_manager.callbacks.on_train_begin()
trainer.train()
compression_manager.callbacks.on_train_end()

from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
save_for_huggingface_upstream(compression_manager.model, tokenizer, training_args.output_dir)
```

# Appendix

## Export to ONNX

Right now, we experimentally support exporting PyTorch model to ONNX model, includes FP32 and INT8 model.

By enabling `--onnx` argument, Intel Neural Compressor will export fp32 ONNX model, INT8 QDQ ONNX model, and INT8 QLinear ONNX model.

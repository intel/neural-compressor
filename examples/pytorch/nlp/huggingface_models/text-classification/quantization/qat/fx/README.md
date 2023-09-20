Step-by-Step
============
This document list steps of reproducing Intel Optimized PyTorch bert models quantization and benchmarking results via Neural Compressor with quantization aware training.
Our example comes from [Huggingface/transformers](https://github.com/huggingface/transformers)

# Prerequisite
## 1. Environment
Python 3.6 or higher version is recommended.
The dependent packages are listed in `requirements.txt`, please install them as follows,
```shell
cd examples/pytorch/nlp/huggingface_models/text-classification/quantization/qat/fx
pip install -r requirements.txt
```
## 2. Prepare fine-tuned model

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
## 1. Enable bert-base-cased/uncased example with the auto QAT strategy of Neural Compressor
  The changes made are as the following:
  * edit run_glue.py:  
    - For quantization, We used neural_compressor in it.  
    - For training, we enabled early stop strategy.  
## 2. To get tuned model and its accuracy: 
```shell
    bash run_quant.sh --input_model=./bert_model  --output_model=./saved_results
```

or

``` shell
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
```
## 3. To get the benchmark of tuned model, including batch_size and throughput: 
```shell
    bash run_benchmark.sh --input_model=./bert_model --config=./saved_results --mode=benchmark --int8=true/false
```

or

```shell
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
        --performance [--int8]
```

# HuggingFace Model Hub
## 1. To upstream into HuggingFace model hub
We provide an API `save_for_huggingface_upstream` to collect configuration files, tokenizer files and INT8 model weights in the format of [transformers](https://github.com/huggingface/transformers). 
```py
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream
save_for_huggingface_upstream(q_model, tokenizer, output_dir)
```
Users can upstream files in the `output_dir` into model hub and reuse them with our `OptimizedModel` API.

## 2. To download from HuggingFace model hub
We provide an API `OptimizedModel` to initialize INT8 models from HuggingFace model hub and its usage is same as the model class provided by [transformers](https://github.com/huggingface/transformers).
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
We also upstreamed several INT8 models into HuggingFace [model hub](https://huggingface.co/models?other=Intel%C2%AE%20Neural%20Compressor) for users to ramp up.

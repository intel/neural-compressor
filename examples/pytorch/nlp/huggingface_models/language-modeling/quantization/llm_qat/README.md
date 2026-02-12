# Quantization Aware Training (QAT)

Quantization-Aware Training (QAT) is a technique designed to bridge the accuracy gap often observed with Post-Training Quantization (PTQ). Unlike PTQ, which applies quantization after model training, QAT simulates the effects of low-precision arithmetic during the training process itself. This allows the model to adapt its weights and activations to quantization constraints, significantly reducing accuracy degradation. As a result, QAT is particularly effective in preserving model performance even at extremely low precisions, such as MXFP8 or MXFP4, making it a critical approach for deploying efficient, high-performance models on resource-constrained hardware.

## Pre-Requisites

Install the requirements for the example:

```bash
pip install -r requirements.txt
```

## Getting Started

In QAT, a model quantized using `prepare_qat()` can be directly fine-tuned with the original training pipeline. During QAT, the scaling factors inside quantizers are frozen and the model weights are fine-tuned.

### Hugging Face QAT

#### QAT

##### Step 1:

Start by training or fine-tuning your model in its original precision (e.g., BF16). This establishes a strong baseline before introducing quantization.

```
accelerate launch --config-file accelerate_config/fsdp1.yaml \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  main.py \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --model_max_length 4096 \
  --dataloader_drop_last True \
  --do_train True \
  --do_eval True \
  --output_dir ./llama3.1-finetuned \
  --dataset Daring-Anteater \
  --num_train_epochs 2.0 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 3000 \
  --eval_strategy steps \
  --eval_steps 3000 \
  --load_best_model_at_end True \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --report_to tensorboard
```

##### Step 2: 

Save the model directly to get a post training quantization model by following this example [auto_round
](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/auto_round/llama3).


```
CUDA_VISIBLE_DEVICES=0 python ../auto_round/llama3/quantize.py  \
    --model_name_or_path ./llama3.1-finetuned  \
    --quantize \
    --dtype MXFP4 \
    --enable_torch_compile \
    --low_gpu_mem_usage \
    --export_format llm_compressor \
    --export_path llama3.1-finetuned-rtn-MXFP4 \
    --iters 0
```

##### Step 3: 

Train/fine-tune the quantized model with a small learning rate, e.g. 1e-5 for Adam optimizer by setting `--quant_scheme MXFP4 --do_train True`

```
accelerate launch --config-file accelerate_config/fsdp1.yaml \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  main.py \
  --model_name_or_path ./llama3.1-finetuned \
  --model_max_length 4096 \
  --dataloader_drop_last True \
  --do_train True \
  --do_eval True \
  --quant_scheme MXFP4 \
  --output_dir ./llama3.1-finetuned-qat \
  --dataset Daring-Anteater \
  --max_steps 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 3000 \
  --eval_strategy steps \
  --eval_steps 3000 \
  --load_best_model_at_end True \
  --save_total_limit 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --report_to tensorboard
```

#### Evaluation

Once QAT is complete, the saved quantized model can be deployed using vLLM for efficient inference. For example, to evaluate on GSM8K:

```
lm_eval \
  --model vllm \
  --model_args pretrained=./llama3.1-finetuned-qat,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.3,max_model_len=32768,enforce_eager=True \
  --tasks gsm8k \
  --batch_size 8
```

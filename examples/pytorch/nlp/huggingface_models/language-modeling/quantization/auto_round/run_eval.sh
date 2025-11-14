

#!/bin/bash
# Check if a model name is passed as an argument, otherwise use the default model path
if [ -z "$1" ]; then
#   model_path="Meta-Llama-3-8B-Instruct-W4A16-G128-AutoRound"
    # model_path="/storage/yiliu7/quantized_model_ds_mxfp8"
    model_path="/storage/yiliu7/quantized_model_ds_mxfp4"
    model_path="/storage/yiliu7/quantized_model_ds_mxfp4"
    model_path="/storage/yiliu7/quantized_model_ds_mxfp8"
    # model_path="qmodels/quantized_model_ds_mxfp8"
    # model_path="./small-qmodels/quantized_model_qwen_mxfp8/"
    # model_path="/storage/yiliu7/quantized_model_qwen_mxfp4"
    # model_path="/storage/yiliu7/quantized_model_qwen_mxfp8"
else
  model_path="$1"
fi

tp_size=8
model_name=$(basename ${model_path})
output_dir="${model_name}-tp${tp_size}-gsm8k-acc"
# task_name="gsm8k"
# batch_size=256
batch_size=512
task_name="piqa,hellaswag,mmlu"
# task_name="mmlu_high_school_biology"

echo "Evaluating model: ${model_path} on task: ${task_name}, output dir: ${output_dir}"
# VLLM_ATTENTION_BACKEND=TRITON_ATTN \
mkdir -p ${output_dir}
# VLLM_ATTENTION_BACKEND=FLASHINFER \

# VLLM_ENABLE_AR_EXT=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_LOGGING_LEVEL=DEBUG  \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_LOGGING_LEVEL=DEBUG  \
# VLLM_ENABLE_V1_MULTIPROCESSING=1  \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False" \
#   --tasks $task_name  \
#     --batch_size 16 \
#     --limit 32 \
#     --log_samples \
#     --seed 42 \
#     --output_path ${output_dir} \
#     --show_config 2>&1 | tee ${output_dir}/log.txt
  
# 


# VLLM_ENABLE_AR_EXT=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_MXFP4_PRE_UNPACK_TO_FP8=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enable_expert_parallel=True" \
#   --tasks $task_name  \
#     --batch_size 16 \
#     --limit 256 \
#     --log_samples \
#     --seed 42 \
#     --output_path ${output_dir} \
#     --show_config 2>&1 | tee ${output_dir}/log.txt
  
# -MXFP4 Evaluation
# /storage/yiliu7/quantized_model_qwen_mxfp4 4x200
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_MXFP4_PRE_UNPACK_TO_FP8=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enable_expert_parallel=True" \
#   --tasks $task_name  \
#     --batch_size 16 \
#     --limit 256 \
#     --log_samples \
#     --seed 42 \
#     --output_path ${output_dir} \
#     --show_config 2>&1 | tee ${output_dir}/log.txt

# -MXFP8 Evaluation
# !!! Please set below knobs strictly for MXFP8 model evaluation !!!
# /storage/yiliu7/quantized_model_qwen_mxfp8 4x200
VLLM_ENABLE_AR_EXT=1 \
VLLM_AR_MXFP4_MODULAR_MOE=0 \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
VLLM_MXFP4_PRE_UNPACK_TO_FP8=0 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False" \
  --tasks $task_name  \
    --batch_size $batch_size \
    --log_samples \
    --seed 42 \
    --output_path ${output_dir} \
    --show_config 2>&1 | tee ${output_dir}/log.txt
  

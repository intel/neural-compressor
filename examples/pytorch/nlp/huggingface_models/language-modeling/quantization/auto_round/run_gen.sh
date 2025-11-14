# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_ENABLE_V1_MULTIPROCESSING=0



model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4/"
model_path="quantized_models/DeepSeek-V2-Lite-Chat-MXFP4"
model_path="quantized_model_qwen_mxfp8"
# model_path="quantized_model_ds_mxfp8"
# model_path="quantized_model_ds_mxfp4"
# model_path="quantized_model_qwen_mxfp4"
# model_path="quantized_model_qwen_mxfp4"
# model_path="quantized_models/Qwen3-235B-A22B-MXFP4"
# model_path="quantized_models/Qwen3-30B-A3B-Base-MXFP4"
model_path="/storage/yiliu7/quantized_model_ds_mxfp8"
model_path="/storage/yiliu7/quantized_model_ds_mxfp4"
# model_path="/storage/yiliu7/quantized_model_qwen_mxfp4"
tp_size=4
# /home/yiliu7/workspace/torchutils/examples

# VLLM_ATTENTION_BACKEND=TRITON_ATTN \
# VLLM_LOGGING_LEVEL=DEBUG  \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# python generate.py \
#     --model ${model_path} \
#     --tensor_parallel_size 8 \
#     --max-tokens 16 \
#     --max-num-seqs 32  \
#     --gpu_memory_utilization 0.9 \
#     --distributed_executor_backend mp
#     # --tensor_parallel_size 4


# VLLM_LOGGING_LEVEL=DEBUG  \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=0 \
# VLLM_ENABLE_STATIC_MOE=1 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# python generate.py \
#     --model ${model_path} \
#     --tensor_parallel_size 4 \
#     --max-tokens 16 \
#     --max-num-seqs 32  \
#     --gpu_memory_utilization 0.9 \
#     --distributed_executor_backend mp \
#     --enforce-eager
#     # --tensor_parallel_size 4
    
#         # --enforce-eager \
#     # --max-model-len 1024 \
# VLLM_LOGGING_LEVEL=DEBUG  \
# model_path="/home/yiliu7/workspace/auto-round/inc_examples/quantized_model_ds_mxfp4"

VLLM_AR_MXFP4_MODULAR_MOE=1 \
VLLM_ENABLE_AR_EXT=1 \
VLLM_MXFP4_PRE_UNPACK_TO_FP8=0 \
VLLM_ENABLE_STATIC_MOE=0 \
VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python generate.py \
    --model ${model_path} \
    --tensor_parallel_size $tp_size \
    --max-tokens 16 \
    --max-num-seqs 32  \
    --gpu_memory_utilization 0.75 \
    --no-enable-prefix-caching \
    --enable_expert_parallel
    # \
    # --enforce-eager
    # --tensor_parallel_size 4
    
        # --enforce-eager \
    # --max-model-len 1024 \
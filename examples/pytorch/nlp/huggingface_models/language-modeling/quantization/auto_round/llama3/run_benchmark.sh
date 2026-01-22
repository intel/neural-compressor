#!/bin/bash

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]

# Parse command line arguments
TASKS="piqa,hellaswag,mmlu_llama,gsm8k_llama"
BATCH_SIZE=64
GPU_MEMORY_UTILIZATION=0.8
KV_CACHE_DTYPE="auto"
ATTN_DTYPE="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --tasks=*)
            TASKS="${1#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            shift
            ;;
        --gpu_memory_utilization=*)
            GPU_MEMORY_UTILIZATION="${1#*=}"
            shift
            ;;
        --static_kv_dtype=*)
            KV_CACHE_DTYPE="${1#*=}"
            shift
            ;;
        --static_attention_dtype=*)
            ATTN_DTYPE="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# for fp8 kv cache
if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    export VLLM_ATTENTION_BACKEND="FLASHINFER"
    echo "Using FP8 for KV cache"
fi

# for fp8 attention cache
if [[ "$ATTN_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
    export VLLM_ATTENTION_BACKEND="FLASHINFER"
    KV_CACHE_DTYPE="fp8"
    echo "Using FP8 Attention"
fi

# Validate required parameters
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]"
    echo "Example: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=Llama-3.1-8B-MXFP8"
    exit 1
fi

# Count available GPUs and set tensor_parallel_size
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    # Count comma-separated GPU IDs
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}
else
    TENSOR_PARALLEL_SIZE=1
fi

echo "Running benchmark with parameters:"
echo "  Model Path: $MODEL_PATH"
echo "  Tasks: $TASKS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check if the model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path '$MODEL_PATH' does not exist!"
    exit 1
fi

# Set common environment variables
export VLLM_ENABLE_AR_EXT=1
export TORCH_COMPILE_DISABLE=1

# Function to run evaluation for specific tasks
run_evaluation() {
    local tasks=$1
    local add_bos_token=$2
    local extra_args=$3
    
    echo "Running evaluation for tasks: $tasks (add_bos_token=$add_bos_token)"
    
    # Print the command being executed
    local cmd="lm_eval --model vllm --model_args pretrained=\"$MODEL_PATH\",add_bos_token=$add_bos_token,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=1,max_model_len=8192,kv_cache_dtype=${KV_CACHE_DTYPE} --tasks $tasks --batch_size $BATCH_SIZE $extra_args"
    echo "Executing command: $cmd"
    
    lm_eval --model vllm \
        --model_args pretrained="$MODEL_PATH",add_bos_token=$add_bos_token,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=1,max_model_len=8192,kv_cache_dtype=${KV_CACHE_DTYPE} \
        --tasks $tasks \
        --batch_size $BATCH_SIZE \
        $extra_args

    if [[ $? -ne 0 ]]; then
        echo "Error: Evaluation failed for tasks: $tasks"
        return 1
    fi
}


# Check if tasks contain gsm8k_llama, mmlu_llama, or longbench
NEED_SPLIT=false
OTHER_TASKS="$TASKS"
SPECIAL_TASKS=""
LONGBENCH_TASK=""

if [[ "$TASKS" == *"gsm8k_llama"* ]]; then
    SPECIAL_TASKS="gsm8k_llama"
    OTHER_TASKS=$(echo "$OTHER_TASKS" | sed 's/,*gsm8k_llama,*//' | sed 's/^,//' | sed 's/,$//')
    NEED_SPLIT=true
fi
if [[ "$TASKS" == *"mmlu_llama"* ]]; then
    if [[ -n "$SPECIAL_TASKS" ]]; then
        SPECIAL_TASKS="$SPECIAL_TASKS,mmlu_llama"
    else
        SPECIAL_TASKS="mmlu_llama"
    fi
    OTHER_TASKS=$(echo "$OTHER_TASKS" | sed 's/,*mmlu_llama,*//' | sed 's/^,//' | sed 's/,$//')
    NEED_SPLIT=true
fi
if [[ "$TASKS" == *"longbench"* ]]; then
    LONGBENCH_TASK="longbench"
    OTHER_TASKS=$(echo "$OTHER_TASKS" | sed 's/,*longbench,*//' | sed 's/^,//' | sed 's/,$//')
    NEED_SPLIT=true
fi

if [[ "$NEED_SPLIT" == true ]]; then
    if [[ -n "$OTHER_TASKS" ]]; then
        echo "Running general tasks"
        run_evaluation "$OTHER_TASKS" true ""
        if [[ $? -ne 0 ]]; then
            echo "Skipping special tasks due to previous failure"
            exit 1
        fi
    fi
    
    # Run special tasks (gsm8k_llama, mmlu_llama)
    if [[ -n "$SPECIAL_TASKS" ]]; then
        IFS=',' read -ra SPECIAL_ARRAY <<< "$SPECIAL_TASKS"
        for special_task in "${SPECIAL_ARRAY[@]}"; do
            echo "Running $special_task with chat template"
            run_evaluation "$special_task" true "--apply_chat_template --fewshot_as_multiturn"
            if [[ $? -ne 0 ]]; then
                echo "Benchmark failed on $special_task!"
                exit 1
            fi
        done
    fi
    
    # Run longbench task with special configuration
    if [[ -n "$LONGBENCH_TASK" ]]; then
        echo "Running longbench with special configuration"
        local longbench_cmd="lm_eval --model vllm --model_args pretrained=\"$MODEL_PATH\",trust_remote_code=True,dtype=bfloat16,max_model_len=66000,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,enable_prefix_caching=False --tasks longbench --seed 42 --batch_size $BATCH_SIZE --apply_chat_template --gen_kwargs '{\"temperature\":0.0}'"
        echo "Executing command: $longbench_cmd"
        
        lm_eval --model vllm \
            --model_args pretrained="$MODEL_PATH",trust_remote_code=True,dtype=bfloat16,max_model_len=66000,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,enable_prefix_caching=False \
            --tasks longbench \
            --seed 42 \
            --batch_size $BATCH_SIZE \
            --apply_chat_template \
            --gen_kwargs '{"temperature":0.0}'
        
        if [[ $? -ne 0 ]]; then
            echo "Benchmark failed on longbench!"
            exit 1
        fi
    fi
else
    run_evaluation "$TASKS" true ""
fi

if [[ $? -eq 0 ]]; then
    echo "Benchmark completed successfully!"
else
    echo "Benchmark failed!"
    exit 1
fi

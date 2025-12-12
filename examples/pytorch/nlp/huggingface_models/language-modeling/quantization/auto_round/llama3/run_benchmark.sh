#!/bin/bash

# Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]

# Parse command line arguments
TASKS="piqa,hellaswag,mmlu,gsm8k"
BATCH_SIZE=512
GPU_MEMORY_UTILIZATION=0.8

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
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

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
    
    echo "Running evaluation for tasks: $tasks (add_bos_token=$add_bos_token)"
    
    # Print the command being executed
    local cmd="lm_eval --model vllm --model_args pretrained=\"$MODEL_PATH\",add_bos_token=$add_bos_token,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=1,max_model_len=8192 --tasks $tasks --batch_size $BATCH_SIZE"
    echo "Executing command: $cmd"
    
    lm_eval --model vllm \
        --model_args pretrained="$MODEL_PATH",add_bos_token=$add_bos_token,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=1,max_model_len=8192 \
        --tasks $tasks \
        --batch_size $BATCH_SIZE
        
    if [[ $? -ne 0 ]]; then
        echo "Error: Evaluation failed for tasks: $tasks"
        return 1
    fi
}

# Check if tasks contain gsm8k (requires add_bos_token=False)
if [[ "$TASKS" == *"gsm8k"* ]]; then
    # If gsm8k is the only task
    if [[ "$TASKS" == "gsm8k" ]]; then
        run_evaluation "$TASKS" false
    else
        # Split tasks: run gsm8k separately with add_bos_token=False
        OTHER_TASKS=$(echo "$TASKS" | sed 's/,*gsm8k,*//' | sed 's/^,//' | sed 's/,$//')
        
        if [[ -n "$OTHER_TASKS" ]]; then
            echo "Running general tasks with add_bos_token=True"
            run_evaluation "$OTHER_TASKS" true
            
            if [[ $? -eq 0 ]]; then
                echo "Running GSM8K with add_bos_token=False"
                run_evaluation "gsm8k" false
            else
                echo "Skipping GSM8K due to previous failure"
                exit 1
            fi
        else
            run_evaluation "gsm8k" false
        fi
    fi
else
    # No gsm8k task, use add_bos_token=True for all tasks
    run_evaluation "$TASKS" true
fi

if [[ $? -eq 0 ]]; then
    echo "Benchmark completed successfully!"
else
    echo "Benchmark failed!"
    exit 1
fi
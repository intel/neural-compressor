#!/bin/bash
set -e

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=<path_to_quantized_model>

MODEL_PATH=""
TASKS="gsm8k,mmlu,piqa,hellaswag"
BATCH_SIZE="auto"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.8
KV_CACHE_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"

usage() {
	echo "Usage: bash run_benchmark.sh --model_path=<path_to_quantized_model> [--tasks=<tasks>] [--batch_size=<size>]"
	echo "  --model_path               Path to the quantized model (required)"
	echo "  --tasks                    Task name(s) to evaluate (default: gsm8k,mmlu,piqa,hellaswag)"
	echo "  --batch_size               Batch size (default: auto)"
	echo "  --max_model_len            Max model length (default: 8192)"
	echo "  --gpu_memory_utilization   GPU memory utilization (default: 0.8)"
	echo "  --static_kv_dtype          Data type for static kv cache (default: auto)"
	echo "  --static_attention_dtype   Data type for static attention cache (default: auto)"
	echo ""
	echo "Examples:"
	echo "  CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=/path/to/model --tasks=gsm8k --batch_size=64"
	exit 1
}

for arg in "$@"; do
	case $arg in
		--model_path=*)
			MODEL_PATH="${arg#*=}"
			;;
		--tasks=*)
			TASKS="${arg#*=}"
			;;
		--batch_size=*)
			BATCH_SIZE="${arg#*=}"
			;;
		--max_model_len=*)
			MAX_MODEL_LEN="${arg#*=}"
			;;
		--gpu_memory_utilization=*)
			GPU_MEMORY_UTILIZATION="${arg#*=}"
			;;
		--static_kv_dtype=*)
			KV_CACHE_DTYPE="${arg#*=}"
			;;
		--static_attention_dtype=*)
			STATIC_ATTENTION_DTYPE="${arg#*=}"
			;;
		-h|--help)
			usage
			;;
		*)
			echo "Unknown parameter: $arg"
			usage
			;;
	esac
done

# for fp8 kv cache
if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
    export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
    echo "Using FP8 for KV cache"
fi

# for fp8 attention cache
if [[ "$STATIC_ATTENTION_DTYPE" == "fp8" ]]; then
    export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
    export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
    KV_CACHE_DTYPE="fp8"
    echo "Using FP8 Attention"
fi


if [[ -z "$MODEL_PATH" ]]; then
	echo "Error: --model_path is required"
	usage
fi

if [[ ! -d "$MODEL_PATH" ]]; then
	echo "Error: Model path '$MODEL_PATH' does not exist!"
	exit 1
fi

# Count available GPUs from CUDA_VISIBLE_DEVICES and set tensor_parallel_size.
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
	IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
	TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}
else
	TENSOR_PARALLEL_SIZE=1
fi

echo "Running Kimi benchmark with parameters:"
echo "  Model Path: $MODEL_PATH"
echo "  Tasks: $TASKS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  KV Cache Dtype: $KV_CACHE_DTYPE"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export VLLM_QDQ=1
export VLLM_MXFP4_USE_MARLIN=1

CMD="lm_eval --model vllm --model_args pretrained=\"$MODEL_PATH\",tensor_parallel_size=$TENSOR_PARALLEL_SIZE,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,kv_cache_dtype=$KV_CACHE_DTYPE,trust_remote_code=True --tasks $TASKS --batch_size $BATCH_SIZE"

echo "Executing command:"
echo "VLLM_QDQ=1 VLLM_MXFP4_USE_MARLIN=1 $CMD"

lm_eval --model vllm \
	--model_args pretrained="$MODEL_PATH",tensor_parallel_size=$TENSOR_PARALLEL_SIZE,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,kv_cache_dtype=$KV_CACHE_DTYPE,trust_remote_code=True \
	--tasks "$TASKS" \
	--batch_size "$BATCH_SIZE"

echo "Benchmark completed successfully!"

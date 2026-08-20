#!/bin/bash
set -e

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh --model_path=<path_to_quantized_model>

MODEL_PATH=""
TASKS="gsm8k,mmlu,piqa,hellaswag"
BATCH_SIZE="auto"
MAX_MODEL_LEN=8192

usage() {
	echo "Usage: bash run_benchmark.sh --model_path=<path_to_quantized_model>"
	echo "Optional: --tasks=<task1,task2> --batch_size=<auto|int> --max_model_len=<int>"
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
		-h|--help)
			usage
			;;
		*)
			echo "Unknown parameter: $arg"
			usage
			;;
	esac
done

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
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export VLLM_QDQ=1

CMD="lm_eval --model vllm --model_args pretrained=\"$MODEL_PATH\",tensor_parallel_size=$TENSOR_PARALLEL_SIZE,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN,trust_remote_code=True --tasks $TASKS --batch_size $BATCH_SIZE"

echo "Executing command:"
echo "VLLM_QDQ=1 $CMD"

lm_eval --model vllm \
	--model_args pretrained="$MODEL_PATH",tensor_parallel_size=$TENSOR_PARALLEL_SIZE,data_parallel_size=1,max_model_len=$MAX_MODEL_LEN,trust_remote_code=True \
	--tasks "$TASKS" \
	--batch_size "$BATCH_SIZE"

echo "Benchmark completed successfully!"

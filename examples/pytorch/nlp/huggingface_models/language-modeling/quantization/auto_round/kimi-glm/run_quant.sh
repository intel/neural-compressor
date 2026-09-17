#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
FORMAT="llm_compressor"
KV_CACHE_DTYPE=""
STATIC_ATTENTION_DTYPE=""
# required transformers==4.57.6 for fp8kv static quant

usage() {
	echo "Usage: bash run_quant.sh --dtype=<mxfp4> --input_model=<path_or_name> --output_model=<output_dir>"
	echo "Optional: --format=<auto_round|llm_compressor> --static_kv_dtype=<dtype> --static_attention_dtype=<dtype>"
	echo ""
	echo "Model type is auto-detected from --input_model name:"
	echo "  - Kimi (contains 'kimi'): MXFP4 with ignore_layers for shared_experts/self_attn/mlp"
	echo "  - GLM  (contains 'glm'):  BF16 base + MXFP4 experts via layer_config"
	exit 1
}

for arg in "$@"; do
	case $arg in
		--dtype=*)
			DTYPE="${arg#*=}"
			;;
		--input_model=*)
			INPUT_MODEL="${arg#*=}"
			;;
		--output_model=*)
			OUTPUT_MODEL="${arg#*=}"
			;;
		--format=*)
			FORMAT="${arg#*=}"
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
			echo "Unknown option: $arg"
			usage
			;;
	esac
done

[[ -z "$DTYPE" ]] && echo "Error: --dtype is required" && usage
[[ -z "$INPUT_MODEL" ]] && echo "Error: --input_model is required" && usage
[[ -z "$OUTPUT_MODEL" ]] && echo "Error: --output_model is required" && usage

# Auto-detect model type from input_model name
MODEL_LOWER=$(echo "$INPUT_MODEL" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" == *kimi* ]]; then
	MODEL_TYPE="kimi"
elif [[ "$MODEL_LOWER" == *glm* ]]; then
	MODEL_TYPE="glm"
else
	echo "Error: Cannot detect model type from '$INPUT_MODEL'. Name must contain 'kimi' or 'glm'."
	exit 1
fi

echo "Detected model type: $MODEL_TYPE"

cd "$SCRIPT_DIR"
QUANTIZE_ARGS=(
	--dtype "$DTYPE" \
	--input_model "$INPUT_MODEL" \
	--output_model "$OUTPUT_MODEL" \
	--format "$FORMAT" \
	--model_type "$MODEL_TYPE"
)

[[ -n "$KV_CACHE_DTYPE" ]] && QUANTIZE_ARGS+=(--static_kv_dtype "$KV_CACHE_DTYPE")
[[ -n "$STATIC_ATTENTION_DTYPE" ]] && QUANTIZE_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")

python quantize.py "${QUANTIZE_ARGS[@]}"


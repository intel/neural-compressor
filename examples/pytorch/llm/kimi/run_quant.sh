#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL="moonshotai/Kimi-K2.6"
OUTPUT_MODEL=""
FORMAT="llm_compressor"
IGNORE_LAYERS="shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj"
KV_CACHE_DTYPE=""
STATIC_ATTENTION_DTYPE=""
# required transformers==4.57.6 for fp8kv static quant

usage() {
	echo "Usage: bash run_quant.sh --dtype=<mxfp4> --input_model=<path_or_name> --output_model=<output_dir>"
	echo "Optional: --format=<auto_round|llm_compressor> --ignore_layers=<comma_separated_patterns>"
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
		--ignore_layers=*)
			IGNORE_LAYERS="${arg#*=}"
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

cd "$SCRIPT_DIR"
QUANTIZE_ARGS=(
	--dtype "$DTYPE" \
	--input_model "$INPUT_MODEL" \
	--output_model "$OUTPUT_MODEL" \
	--format "$FORMAT" \
	--ignore_layers "$IGNORE_LAYERS"
)

[[ -n "$KV_CACHE_DTYPE" ]] && QUANTIZE_ARGS+=(--static_kv_dtype "$KV_CACHE_DTYPE")
[[ -n "$STATIC_ATTENTION_DTYPE" ]] && QUANTIZE_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")

python quantize.py "${QUANTIZE_ARGS[@]}"
	

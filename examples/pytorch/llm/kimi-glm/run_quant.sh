#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DTYPE=""
INPUT_MODEL=""
OUTPUT_MODEL=""
EXPORT_FORMAT="llm_compressor"
STATIC_KV_DTYPE="auto"
STATIC_ATTENTION_DTYPE="auto"
# required transformers==4.57.6 for fp8kv static quant

usage() {
	echo "Usage: bash run_quant.sh --dtype=<dtype> --input_model=<input_model> --output_model=<output_model>"
	echo "  --dtype                    quantization data type (currently: mxfp4)"
	echo "  --input_model              Hugging Face model ID or local path"
	echo "  --output_model             output directory for the quantized model"
	echo "  --export_format            export format (default: llm_compressor)"
	echo "  --static_kv_dtype          data type for static kv cache (default: auto)"
	echo "  --static_attention_dtype   data type for static attention cache (default: auto)"
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
		--export_format=*)
			EXPORT_FORMAT="${arg#*=}"
			;;
		--static_kv_dtype=*)
			STATIC_KV_DTYPE="${arg#*=}"
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

echo "Starting quantization with parameters:"
echo "  Model Type: $MODEL_TYPE"
echo "  Data Type: $DTYPE"
echo "  Input Model: $INPUT_MODEL"
echo "  Output Model: $OUTPUT_MODEL"

cd "$SCRIPT_DIR"
QUANTIZE_ARGS=(
	--dtype "$DTYPE"
	--model_name_or_path "$INPUT_MODEL"
	--export_path "$OUTPUT_MODEL"
	--export_format "$EXPORT_FORMAT"
	--model_type "$MODEL_TYPE"
)

if [[ "$STATIC_KV_DTYPE" != "auto" ]]; then
	QUANTIZE_ARGS+=(--static_kv_dtype "$STATIC_KV_DTYPE")
fi
if [[ "$STATIC_ATTENTION_DTYPE" != "auto" ]]; then
	QUANTIZE_ARGS+=(--static_attention_dtype "$STATIC_ATTENTION_DTYPE")
fi

python quantize.py "${QUANTIZE_ARGS[@]}"

echo "Quantization completed successfully!"
echo "Output model saved to: $OUTPUT_MODEL"


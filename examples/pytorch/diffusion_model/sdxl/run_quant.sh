#!/bin/bash
set -euo pipefail
set -x

input_model="stabilityai/stable-diffusion-xl-base-1.0"
output_model="./saved_results"

for var in "$@"; do
    case "${var}" in
        --input_model=*) input_model="${var#*=}" ;;
        --output_model=*) output_model="${var#*=}" ;;
        *) echo "Error: No such parameter: ${var}"; exit 1 ;;
    esac
done

python3 main.py \
    --model "${input_model}" \
    --output_dir "${output_model}" \
    --scheme MXFP8 \
    --disable_opt_rtn \
    --quantize

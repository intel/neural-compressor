#!/bin/bash
set -euo pipefail
set -x

input_model="stabilityai/stable-diffusion-xl-base-1.0"
quantized_model=""
dataset_location="captions_source.tsv"
output_image_path="./tmp_imgs"
limit=-1
num_inference_steps=50
guidance_scale=7.5
seed=42

for var in "$@"; do
    case "${var}" in
        --input_model=*) input_model="${var#*=}" ;;
        --quantized_model=*) quantized_model="${var#*=}" ;;
        --dataset_location=*) dataset_location="${var#*=}" ;;
        --output_image_path=*) output_image_path="${var#*=}" ;;
        --limit=*) limit="${var#*=}" ;;
        --num_inference_steps=*) num_inference_steps="${var#*=}" ;;
        --guidance_scale=*) guidance_scale="${var#*=}" ;;
        --seed=*) seed="${var#*=}" ;;
        *) echo "Error: No such parameter: ${var}"; exit 1 ;;
    esac
done

scheme="BF16"
model_args=()
if [[ -n "${quantized_model}" ]]; then
    scheme="MXFP8"
    model_args=(--quantized_model_path "${quantized_model}")
fi

common_args=(
    --model "${input_model}"
    --scheme "${scheme}"
    --output_image_path "${output_image_path}"
    --num_inference_steps "${num_inference_steps}"
    --guidance_scale "${guidance_scale}"
    --seed "${seed}"
    --limit "${limit}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    visible_gpus=${#gpu_ids[@]}
    subset_dir=$(mktemp -d)
    trap 'rm -rf "${subset_dir}"' EXIT

    python3 dataset_split.py \
        --split_num "${visible_gpus}" \
        --input_file "${dataset_location}" \
        --output_file "${subset_dir}/subset" \
        --limit "${limit}"

    pids=()
    for ((index=0; index<visible_gpus; index++)); do
        CUDA_VISIBLE_DEVICES="${gpu_ids[index]}" python3 main.py \
            "${common_args[@]}" \
            "${model_args[@]}" \
            --eval_dataset "${subset_dir}/subset_${index}.tsv" \
            --inference &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    [[ "${status}" -eq 0 ]] || exit "${status}"
else
    python3 main.py \
        "${common_args[@]}" \
        "${model_args[@]}" \
        --eval_dataset "${dataset_location}" \
        --inference
fi

python3 main.py \
    --eval_dataset "${dataset_location}" \
    --output_image_path "${output_image_path}" \
    --limit "${limit}" \
    --accuracy

#!/bin/bash
set -ex

function main {
  init_params "$@"
  run_benchmark
}

function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --limit=*)
          limit=$(echo $var |cut -f2 -d=)
      ;;
      --output_image_path=*)
          output_image_path=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done
}

function run_benchmark {
    dataset_location=${dataset_location:="drawbench.tsv"}
    limit=${limit:=-1}
    output_image_path=${output_image_path:="./drawbench_images"}

    if [ "${topology}" = "zimage_mxfp8" ] || [ "${topology}" = "zimage_bf16" ]; then
        extra_cmd="--inference --backend vllm_omni"
    else
        echo "Unknown topology: ${topology} (use zimage_mxfp8 or zimage_bf16)"
        exit 1
    fi

    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        gpu_list="${CUDA_VISIBLE_DEVICES:-}"
        IFS=',' read -ra gpu_ids <<< "$gpu_list"
        visible_gpus=${#gpu_ids[@]}
        echo "visible_gpus: ${visible_gpus}"

        python dataset_split.py --split_num ${visible_gpus} \
            --input_file ${dataset_location} --limit ${limit} \
            --output_file subset

        program_pid=()
        for ((i=0; i<visible_gpus; i++)); do
            export CUDA_VISIBLE_DEVICES=${gpu_ids[i]}
            python3 main.py \
                --model ${input_model} \
                --output_image_path ${output_image_path} \
                --eval_dataset "subset_$i.tsv" \
                ${extra_cmd} &
            program_pid+=($!)
            echo "Start (PID: ${program_pid[-1]}, GPU: ${i})"
        done
        failed=0
        for pid in "${program_pid[@]}"; do
            if ! wait "$pid"; then
                echo "Generation failed (PID: ${pid})" >&2
                failed=1
            fi
        done
        if [ "$failed" -ne 0 ]; then
            return 1
        fi
    else
        python3 main.py \
            --model ${input_model} \
            --output_image_path ${output_image_path} \
            --eval_dataset ${dataset_location} \
            --limit ${limit} \
            ${extra_cmd}
    fi

    echo "Start calculating final score..."
    python3 main.py --accuracy \
        --output_image_path ${output_image_path} \
        --eval_dataset ${dataset_location} \
        --limit ${limit}
}

main "$@"

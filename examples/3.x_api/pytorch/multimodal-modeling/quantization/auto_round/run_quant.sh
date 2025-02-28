#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
       --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
       ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=''
    batch_size=8
    DATASET_NAME="NeelNanda/pile-10k"
    tuned_checkpoint="saved_results"

    if [ "${topology}" = "phi3_vlm_128k_autoround_int4" ]; then
        model_name_or_path="microsoft/Phi-3-vision-128k-instruct"
    fi

    python -u mllm.py \
        --model ${model_name_or_path} \
        --dataset ${DATASET_NAME} \
        --quantize \
        --iters ${iters} \
        --output_dir ${tuned_checkpoint} \
        --batch_size ${batch_size} \
        ${extra_cmd}
}

main "$@"

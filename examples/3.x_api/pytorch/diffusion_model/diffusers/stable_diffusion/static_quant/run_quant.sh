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
    DATASET_DIR=${dataset_location}
    tuned_checkpoint="unet_quantized_model.pt2"

    if [ "${topology}" = "sd21_static_int8" ]; then
        model_name_or_path="stabilityai/stable-diffusion-2-1"
    elif [ "${topology}" = "lcm_static_int8" ]; then
        model_name_or_path="SimianLuo/LCM_Dreamshaper_v7"
    else
        echo "Error: No such topology: ${topology}"
        exit 1
    fi

    python main.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_path=${DATASET_DIR} \
        --quantized_model_path=${tuned_checkpoint} \
        --compile_inductor \
        --precision=int8-bf16 \
        --calibration
}

main "$@"

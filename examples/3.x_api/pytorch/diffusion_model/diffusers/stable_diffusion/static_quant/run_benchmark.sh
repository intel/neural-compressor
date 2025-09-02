#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=10
  batch_size=8
  tuned_checkpoint=saved_results
  echo ${max_eval_samples}
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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --optimized=*)
          optimized=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd=''
    mode_cmd=''
    DATASET_DIR=${dataset_location}
    tuned_checkpoint="unet_quantized_model.pt2"

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --benchmark -w 1 -i ${iters} "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${optimized} == "true" ]]; then
        extra_cmd=$extra_cmd" --quantized_model_path=${tuned_checkpoint}  --precision=int8-bf16 "
    else
        extra_cmd=$extra_cmd" --precision=bf16 "
    fi
    echo $extra_cmd

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
        --compile_inductor \
        ${extra_cmd} ${mode_cmd}
        
}

main "$@"

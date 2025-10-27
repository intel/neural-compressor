#!/bin/bash
set -x

function main {
0
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16 
  tuned_checkpoint=saved_results
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
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
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
    extra_cmd=""

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --benchmark"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi


    if [ "${topology}" = "pokemon_diffusers" ]; then
        model_name_or_path="lambdalabs/sd-pokemon-diffusers"
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    python -u ../../run_diffusion.py \
       --model_name_or_path ${model_name_or_path} \
       --output_dir ${tuned_checkpoint} \
       --base_images ../../base_images \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"

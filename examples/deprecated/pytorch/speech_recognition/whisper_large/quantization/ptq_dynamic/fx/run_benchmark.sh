#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark
}

# init params
function init_params {
  iters=100
  batch_size=1
  output_model=saved_results
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
          output_model=$(echo $var |cut -f2 -d=)
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;

    esac
  done

}

# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iters ${iters} --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
    extra_cmd=""

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi

    python run_whisper_large.py \
            --output_dir ${output_model} \
            --batch_size $batch_size \
	    --cache_dir ${dataset_location} \
            ${mode_cmd} \
            ${extra_cmd}

}

main "$@"

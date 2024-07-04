#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  batch_size=32
  iters=100

  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo $var |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    python main.py \
            --input-graph ${input_model} \
            --mode ${mode} \
            --dataset_location ${dataset_location} \
            --batch_size ${batch_size} \
            --benchmark \
            --iters ${iters} \
            ${extra_cmd}
}

main "$@"

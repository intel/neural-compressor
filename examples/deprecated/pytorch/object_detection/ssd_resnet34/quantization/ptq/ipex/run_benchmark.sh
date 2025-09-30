#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  batch_size=16
  iters=100
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
        mode_cmd="--accuracy-mode "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd="--benchmark "
        extra_cmd=$extra_cmd" --iteration ${iters}"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi

    python infer.py \
        --data ${dataset_location} \
        --device 0 \
        --checkpoint  ${input_model} \
        -w 10 \
        -j 0 \
        --no-cuda \
        --batch-size ${batch_size} \
        --accuracy-mode \
        ${extra_cmd}
}

main "$@"

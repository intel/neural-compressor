#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  tuned_checkpoint=saved_results
  batch_size=30
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
    python setup.py install
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iterations ${iters} --performance "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [[ ${int8} == "true" ]]; then
        extra_cmd="--int8"
    fi

    python run_eval.py \
        --data ${dataset_location} \
        -a ${input_model} \
        -b ${batch_size} \
        -j 1 \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"

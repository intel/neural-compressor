#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  tuned_checkpoint=./saved_results
  for var in "$@"
  do
    case $var in
      --tuned_checkpoint=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      --voc_dataset_location=*)
          voc_dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
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
    if [[ ${mode} == "dice" ]]; then
        mode_cmd=" --dice=True"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iter ${iters} --performance=True "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd="--int8 ${dataset_location}"
    else
        extra_cmd="${dataset_location}"
    fi

    python main.py \
            --tuned_checkpoint ${tuned_checkpoint} \
            ${mode_cmd} \
            ${extra_cmd}
}

main "$@"
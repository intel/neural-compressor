#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  batch_size=16
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

    python infer.py \
        --data ${dataset_location} \
        --device 0 \
        --checkpoint  ${input_model} \
        -w 10 \
        -j 0 \
        --no-cuda \
        --batch-size ${batch_size} \
        --tune \
        --accuracy-mode \

}

main "$@"


#!/bin/bash
set -x

function main {

  init_params "$@"

  run_tuning

}

# init params
function init_params {
  anno_path="./label_map.yaml"
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --anno_path=*)
          anno_path=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
  python  main.py \
          --input-graph "${input_model}" \
          --output_model "${output_model}" \
          --dataset_location "${dataset_location}" \
          --anno_path "${anno_path}" \
          --tune
}

main "$@"

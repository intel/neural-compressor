#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  batch_size=1
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {
  python main.py \
    --input_model ${input_model} \
    --mode ${mode} \
    --batch_size ${batch_size} \
    --benchmark \
    --output_dir "./outputs"

}

main "$@"

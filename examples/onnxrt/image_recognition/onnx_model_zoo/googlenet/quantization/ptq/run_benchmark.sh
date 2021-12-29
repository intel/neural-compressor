#!/bin/bash
set -x

function main {
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --data_path=*)
          data_path=$(echo $var |cut -f2 -d=)
      ;;
      --label_path=*)
          label_path=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    python main.py \
            --model_path ${input_model} \
            --data_path ${data_path} \
            --label_path ${label_path} \
            --config ${config} \
            --mode=${mode} \
            --benchmark
            
}

main "$@"

#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    python main.py \
            --model_name_or_path microsoft/codebert-base \
            --model_path ${input_model} \
            --data_path ${dataset_location} \
            --output_model ${output_model} \
            --tune

}

main "$@"

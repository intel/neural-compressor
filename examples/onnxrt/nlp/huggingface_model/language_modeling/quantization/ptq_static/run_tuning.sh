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
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [[ "${input_model}" =~ "gpt2" ]]; then
        model_name_or_path="gpt2"
    fi
    if [[ "${input_model}" =~ "distilgpt2" ]]; then
        model_name_or_path="distilgpt2"
    fi
    
    python main.py --model_path ${input_model} \
                  --data_path ${dataset_location} \
                  --model_name_or_path ${model_name_or_path} \
                  --output_model ${output_model} \
                  --tune 
}

main "$@"
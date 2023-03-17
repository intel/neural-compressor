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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    model_name_or_path="bert-base-uncased"
    batch_size=8
    task_name="mrpc"
    model_type="bert"

    python main.py \
           --model_path ${input_model} \
           --output_model ${output_model} \
           --model_name_or_path ${model_name_or_path} \
           --data_path ${dataset_location} \
           --task ${task_name} \
           --batch_size ${batch_size} \
           --tune
}

main "$@"




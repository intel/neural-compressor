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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
      dynamic_length=False
    elif [[ ${mode} == "performance" ]]; then
      dynamic_length=True
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi

    model_name_or_path="bert-base-uncased"
    task_name="mrpc"
    model_type="bert"

    python main.py \
           --model_path ${input_model} \
           --model_name_or_path ${model_name_or_path} \
           --data_path ${dataset_location} \
           --task ${task_name} \
           --batch_size ${batch_size} \
           --mode ${mode} \
           --dynamic_length ${dynamic_length} \
           --benchmark
            
}

main "$@"


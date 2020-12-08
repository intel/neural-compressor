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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          data_dir=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --config=*)
          bert_yaml=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    if [ "${topology}" = "bert_base_MRPC" ];then
      task_name='mrpc'
      model_name_or_path='bert-base-uncased'
    fi 

    python bert_base.py --model_path ${input_model} --data_dir ${data_dir} --task_name ${task_name} --input_dir ${model_name_or_path} --tune --config ${bert_yaml} --output_model ${output_model}
}

main "$@"




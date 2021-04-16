#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  bert_yaml="./roberta.yaml"
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
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [ "${topology}" = "roberta_base_MRPC" ];then
      task_name='mrpc'
      model_name_or_path='roberta-base'
    fi 
    python roberta.py --model_path ${input_model} --data_dir ${data_dir} \
    --task_name ${task_name} --input_dir ${model_name_or_path} \
    --tune --config ${bert_yaml} --output_model ${output_model} \
    --model_name_or_path ${model_name_or_path}
}

main "$@"




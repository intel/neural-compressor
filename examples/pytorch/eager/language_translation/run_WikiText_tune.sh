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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --train_data_location=*)
          train_data_location=$(echo $var |cut -f2 -d=)
      ;;
      --eval_data_location=*)
          eval_data_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
       --output_model=*)
           output_model=$(echo $var |cut -f2 -d=)
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
    extra_cmd=''
    SCRIPTS=examples/run_lm_tune.py

    if [ "${topology}" = "gpt" ];then
        model_type='openai-gpt'
        model_name_or_path='openai-gpt'
        config='openai-gpt.yaml'
    elif [ "${topology}" = "gpt2" ]; then
        model_type='gpt2'
        model_name_or_path='gpt2'
        config='gpt2.yaml'
    elif [ "${topology}" = "ctrl" ]; then
        model_type='ctrl'
        model_name_or_path='ctrl'
        config='ctrl.yaml'
    fi

    python $SCRIPTS \
        --model_type ${model_type} \
        --model_name_or_path ${model_name_or_path} \
        --do_eval \
        --train_data_file ${train_data_location} \
        --eval_data_file ${eval_data_location} \
        --config ${config} \
        --no_cuda \
        --output_dir ${input_model} \
        --tune \
        ${extra_cmd}

}

main "$@"

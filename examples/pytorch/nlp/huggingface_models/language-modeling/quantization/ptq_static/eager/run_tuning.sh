#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
       --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
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
    batch_size=16
    model_type='bert'
    approach='post_training_static_quant'

    if [ "${topology}" = "dialogpt_wikitext" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=$input_model 
        model_type='dialogpt'
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
    elif [ "${topology}" = "reformer_crime_and_punishment" ]; then
        TASK_NAME='crime_and_punish'
        model_name_or_path=$input_model 
        model_type='reformer'
    elif [ "${topology}" = "ctrl_WikiText" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=$input_model
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
        sed -i "/relative:/s|relative:.*|relative: 0.05|g" conf.yaml
        
    fi

    sed -i "/: bert/s|name:.*|name: $model_type|g" conf.yaml
    sed -i "/approach:/s|approach:.*|approach: $approach|g" conf.yaml

    python -u run_clm_tune.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${TASK_NAME} \
        --do_eval \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${input_model} \
        --tune \
        ${extra_cmd}

}

main "$@"

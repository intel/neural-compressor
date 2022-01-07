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
    approach='post_training_dynamic_quant'

    if [ "${topology}" = "t5_WMT_en_ro" ];then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path=$input_model
        model_type='t5'
    elif [ "${topology}" = "marianmt_WMT_en_ro" ];then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path=$input_model
        model_type='marianmt'
    elif [ "${topology}" = "pegasus_billsum" ]; then
        TASK_NAME='summarization_billsum'
        model_name_or_path=$input_model 
        model_type='pegasus'
        extra_cmd='--predict_with_generate --max_source_length 1024 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256'
    fi

    sed -i "/: bert/s|name:.*|name: $model_type|g" conf.yaml
    sed -i "/approach:/s|approach:.*|approach: $approach|g" conf.yaml

    python -u run_seq2seq_tune.py \
        --model_name_or_path ${model_name_or_path} \
        --data_dir ${dataset_location} \
        --task ${TASK_NAME} \
        --do_eval \
        --do_train \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --tune \
        ${extra_cmd}
}

main "$@"

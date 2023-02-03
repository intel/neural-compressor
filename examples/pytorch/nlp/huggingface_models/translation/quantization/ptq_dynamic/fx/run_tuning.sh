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
    extra_cmd='None'
    batch_size=16
    model_type='bert'

    if [ "${topology}" = "t5_WMT_en_ro" ];then
        model_name_or_path='t5-small'
        model_type='t5'
        extra_cmd='translate English to Romanian: '
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
        model_type='marianmt'
    fi

    python -u run_translation.py \
        --model_name_or_path ${model_name_or_path} \
        --do_train \
        --do_eval \
        --do_train \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --source_lang en \
        --target_lang ro \
        --dataset_name wmt16 \
        --dataset_config_name ro-en\
        --tune \
        --overwrite_output_dir \
        --source_prefix "$extra_cmd"
}

main "$@"

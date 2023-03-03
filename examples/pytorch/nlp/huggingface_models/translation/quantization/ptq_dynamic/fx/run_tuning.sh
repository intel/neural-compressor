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
    batch_size=16

if [ "${topology}" = "t5_WMT_en_ro" ];then
        model_name_or_path='aretw0/t5-small-finetuned-en-to-ro-dataset_20'
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
    fi

    python -u run_translation.py \
        --model_name_or_path ${model_name_or_path} \
        --do_train \
        --do_eval \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --source_lang en \
        --target_lang ro \
        --dataset_config_name ro-en \
        --dataset_name wmt16 \
        --tune \
        --overwrite_output_dir
}

main "$@"

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
    extra_cmd=''

    if [ "${topology}" = "t5-small" ];then
        source_prefix=`--source_prefix 'translate English to Romanian: ' `
        extra_cmd="--model_name_or_path ${input_model} ${source_prefix}"
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        extra_cmd="--model_name_or_path Helsinki-NLP/opus-mt-en-ro"
    fi

    python -u run_translation.py \
        --do_train \
        --do_eval \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --source_lang en \
        --target_lang ro \
        --dataset_config_name ro-en \
        --dataset_name wmt16 \
        --overwrite_cache \
        --tune \
        --overwrite_output_dir \
        ${extra_cmd}
}

main "$@"

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

    if [ "${topology}" = "pegasus_samsum" ]; then
        model_name_or_path='lvwerra/pegasus-samsum'
        extra_cmd="--dataset_name samsum"
    fi

    python -u run_summarization.py \
        --model_name_or_path ${model_name_or_path} \
        --do_train \
        --do_eval \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        --tune \
        $extra_cmd
}

main "$@"

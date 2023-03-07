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
    batch_size=8
    approach='post_training_static_quant'

    if [ "${topology}" = "reformer_crime_and_punishment" ]; then
        TASK_NAME='crime_and_punish'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "gpt_j_wikitext" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=${input_model}
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
    elif [ "${topology}" = "gpt_neox" ]; then
        TASK_NAME='oscar'
        model_name_or_path=${input_model}
        extra_cmd='--dataset_config_name=unshuffled_original_ast'
    fi

    python -u run_clm.py \
        --model_name_or_path ${input_model} \
        --dataset_name ${TASK_NAME} \
        --do_eval \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        --tune \
        ${extra_cmd}

}

main "$@"

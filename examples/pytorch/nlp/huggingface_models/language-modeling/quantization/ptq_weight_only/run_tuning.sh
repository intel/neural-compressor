#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  weight_only_bits=8
  weight_only_group=-1
  weight_only_scheme=sym
  weight_only_algorithm=RTN
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
       --weight_only_bits=*)
           weight_only_bits=$(echo $var |cut -f2 -d=)
       ;;
       --weight_only_group=*)
           weight_only_group=$(echo $var |cut -f2 -d=)
       ;;
       --weight_only_scheme=*)
           weight_only_scheme=$(echo $var |cut -f2 -d=)
       ;;
       --weight_only_algorithm=*)
           weight_only_algorithm=$(echo $var |cut -f2 -d=)
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

    if [ "${topology}" = "gpt_j_wikitext_weight_only" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=${input_model}
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
    fi

    python -u run_clm.py \
        --model_name_or_path ${input_model} \
        --dataset_name ${TASK_NAME} \
        --do_eval \
        --do_train \
        --per_device_eval_batch_size ${batch_size} \
        --weight_only_bits ${weight_only_bits} \
        --weight_only_group ${weight_only_group} \
        --weight_only_scheme ${weight_only_scheme} \
        --weight_only_algorithm ${weight_only_algorithm} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        --tune \
        ${extra_cmd}

}

main "$@"

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
    MAX_SEQ_LENGTH=128
    model_type='bert'
    approach='post_training_static_quant'

    if [ "${topology}" = "xlm-roberta-base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlm-roberta'
    elif [ "${topology}" = "flaubert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='flaubert'
    elif [ "${topology}" = "barthez_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='barthez'
    elif [ "${topology}" = "longformer_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='longformer'
    elif [ "${topology}" = "layoutlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='layoutlm'
    elif [ "${topology}" = "deberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='deberta'
    elif [ "${topology}" = "squeezebert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='squeezebert'
    elif [ "${topology}" = "xlnet_base_cased_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlnet'
    elif [ "${topology}" = "roberta_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='roberta'
    elif [ "${topology}" = "camembert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='camembert'
    fi

    sed -i "/: bert/s|name:.*|name: $model_type|g" conf.yaml
    sed -i "/approach:/s|approach:.*|approach: $approach|g" conf.yaml

    python -u ./run_glue_tune.py \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_gpu_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ${tuned_checkpoint} \
        --tune \
        ${extra_cmd}
}

main "$@"

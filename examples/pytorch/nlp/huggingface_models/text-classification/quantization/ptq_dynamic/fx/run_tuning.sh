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
    approach='post_training_dynamic_quant'
    TASK_NAME='rte'
    model_name_or_path=${input_model}
    
    if [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='rte'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "distilbert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='distilbert'
    elif [ "${topology}" = "albert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='albert'
    elif [ "${topology}" = "funnel_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='funnel'
    elif [ "${topology}" = "mbart_WNLI" ]; then
        TASK_NAME='WNLI'
        model_name_or_path=$input_model 
        model_type='mbart'
    elif [ "${topology}" = "transfo_xl_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='transfo-xl-wt103'
    elif [ "${topology}" = "ctrl_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='ctrl'
    elif [ "${topology}" = "xlm_roberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlm'
    fi

    python -u ./run_glue.py \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_train \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ${tuned_checkpoint} \
        --tune \
        --overwrite_output_dir \
        ${extra_cmd}
}

main "$@"

#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=int8_model_dir
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
    TASK_NAME='mrpc'
    model_name_or_path=${input_model}
    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_CoLA" ]; then
        TASK_NAME='cola'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_STS-B" ]; then
        TASK_NAME='stsb'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_SST-2" ]; then
        TASK_NAME='sst2'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_RTE" ]; then
        TASK_NAME='rte'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_MRPC" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_QNLI" ]; then
        TASK_NAME='qnli'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='rte'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_CoLA" ]; then
        TASK_NAME='cola'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "funnel_MRPC_fx" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "distilbert_base_MRPC_fx" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "xlm-roberta-base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "flaubert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "barthez_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "longformer_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "layoutlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
        model_type='layoutlm'
    elif [ "${topology}" = "deberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "squeezebert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "xlnet_base_cased_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "roberta_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "camembert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    fi

    python -u ./run_glue.py \
        --model_name_or_path ${input_model} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_train \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ${tuned_checkpoint} \
        --tune \
        --onnx \
        --overwrite_output_dir \
        ${extra_cmd}
}

main "$@"

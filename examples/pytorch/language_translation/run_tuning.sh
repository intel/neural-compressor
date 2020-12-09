#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {

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
           output_model=$(echo $var |cut -f2 -d=)
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
    SCRIPTS=examples/run_glue_tune.py
    MAX_SEQ_LENGTH=128
    model_type='bert'

    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "distilbert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='distilbert-base-uncased'
        model_type='distilbert'
    elif [ "${topology}" = "bert_base_CoLA" ]; then
        TASK_NAME='CoLA'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "bert_base_STS-B" ]; then
        TASK_NAME='STS-B'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "bert_base_SST-2" ]; then
        TASK_NAME='SST-2'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "bert_base_RTE" ]; then
        TASK_NAME='RTE'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "bert_large_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='bert-large-uncased-whole-word-masking'
    elif [ "${topology}" = "bert_large_SQuAD" ]; then
        SCRIPTS=examples/run_squad_tune.py
        TASK_NAME='SQuAD'
        MAX_SEQ_LENGTH=384
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--doc_stride 128'
    elif [ "${topology}" = "bert_large_QNLI" ]; then
        TASK_NAME='QNLI'
        model_name_or_path='bert-large-uncased-whole-word-masking'
    elif [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='RTE'
        model_name_or_path='bert-large-uncased-whole-word-masking'
    elif [ "${topology}" = "bert_large_CoLA" ]; then
        TASK_NAME='CoLA'
        model_name_or_path='bert-large-uncased-whole-word-masking'
    fi

    python -u $SCRIPTS \
        --model_type ${model_type} \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_lower_case \
        --data_dir ${dataset_location} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_gpu_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ${input_model} \
        --tune \
        ${extra_cmd}

}

main "$@"

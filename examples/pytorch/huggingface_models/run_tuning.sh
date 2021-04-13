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
    SCRIPTS=examples/text-classification/run_glue_tune.py
    MAX_SEQ_LENGTH=128
    model_type='bert'

    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='bert'
    elif [ "${topology}" = "distilbert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='distilbert-base-uncased'
        model_type='distilbert'
    elif [ "${topology}" = "albert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='albert'
    elif [ "${topology}" = "funnel_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='funnel'
    elif [ "${topology}" = "bart_WNLI" ]; then
        TASK_NAME='WNLI'
        model_name_or_path=$input_model 
        model_type='bart'
    elif [ "${topology}" = "mbart_WNLI" ]; then
        TASK_NAME='WNLI'
        model_name_or_path=$input_model 
        model_type='mbart'
    elif [ "${topology}" = "t5_WMT_en_ro" ];then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path=$input_model
        model_type='t5'
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
        model_type='marianmt'
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
    elif [ "${topology}" = "pegasus_billsum" ]; then
        TASK_NAME='summarization_billsum'
        model_name_or_path=$input_model 
        model_type='pegasus'
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
        extra_cmd='--predict_with_generate --max_source_length 1024 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256'
    fi

    sed -i "/name:/s|name:.*|name: $model_type|g" conf.yaml

    if [ "${SCRIPTS}" = "examples/text-classification/run_glue_tune.py" ];then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_gpu_eval_batch_size ${batch_size} \
            --no_cuda \
            --output_dir ${input_model} \
            --tune \
            ${extra_cmd}
    elif [ "${SCRIPTS}" = "examples/seq2seq/run_seq2seq_tune.py" ]; then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_name_or_path ${model_name_or_path} \
            --data_dir ${dataset_location} \
            --task ${TASK_NAME} \
            --do_eval \
            --predict_with_generate \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${input_model} \
            --tune \
            ${extra_cmd}
    fi

}

main "$@"

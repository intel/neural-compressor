#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16
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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd=''
    SCRIPTS=examples/text-classification/run_glue_tune.py
    MAX_SEQ_LENGTH=128
    model_type='bert'

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path='bert-base-uncased'
    elif [ "${topology}" = "distilbert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='distilbert-base-uncased'
    elif [ "${topology}" = "albert_base_MRPC" ]; then
        MAX_SEQ_LENGTH=512
        TASK_NAME='MRPC'
        model_name_or_path=$input_model #'albert-base-v1'
    elif [ "${topology}" = "funnel_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model #'funnel-transformer/small'
    elif [ "${topology}" = "bart_WNLI" ]; then
        TASK_NAME='WNLI'
        model_name_or_path=$input_model #'facebook/bart-large'
    elif [ "${topology}" = "mbart_WNLI" ]; then
        TASK_NAME='WNLI'
        model_name_or_path=$input_model #'facebook/mbart-large-cc25'
    elif [ "${topology}" = "xlm_roberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "xlnet_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "gpt2_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "t5_WMT_en_ro" ];then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path=$input_model
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
    elif [ "${topology}" = "marianmt_WMT_en_ro" ]; then
        TASK_NAME='translation_en_to_ro'
        model_name_or_path='Helsinki-NLP/opus-mt-en-ro'
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
    elif [ "${topology}" = "pegasus_billsum" ]; then
        TASK_NAME='summarization_billsum'
        model_name_or_path=$input_model #'google/pegasus-billsum'
        SCRIPTS=examples/seq2seq/run_seq2seq_tune.py
        extra_cmd='--predict_with_generate --max_source_length 1024 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256'
    elif [ "${topology}" = "dialogpt_wikitext" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=$input_model 
        SCRIPTS=examples/language-modeling/run_clm_tune.py
        approach="post_training_static_quant"
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
    elif [ "${topology}" = "reformer_crime_and_punishment" ]; then
        TASK_NAME='crime_and_punish'
        model_name_or_path=$input_model 
        SCRIPTS=examples/language-modeling/run_clm_tune.py
        approach="post_training_static_quant"
    elif [ "${topology}" = "xlm-roberta-base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "flaubert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "barthez_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "longformer_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "layoutlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "deberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    elif [ "${topology}" = "squeezebert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

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
            --iters ${iters} \
            ${mode_cmd} \
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
            --iters ${iters} \
            ${mode_cmd} \
            ${extra_cmd}
    elif [ "${SCRIPTS}" = "examples/language-modeling/run_clm_tune.py" ]; then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${TASK_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${input_model} \
            --iters ${iters} \
            ${mode_cmd} \
            ${extra_cmd}
    fi

}

main "$@"

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
    approach='post_training_dynamic_quant'

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
    elif [ "${topology}" = "xlm_roberta_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlm_roberta'
    elif [ "${topology}" = "xlnet_base_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlnet_base_cased'
    elif [ "${topology}" = "gpt2_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='gpt2'
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
    elif [ "${topology}" = "dialogpt_wikitext" ]; then
        TASK_NAME='wikitext'
        model_name_or_path=$input_model 
        model_type='dialogpt'
        SCRIPTS=examples/language-modeling/run_clm_tune.py
        approach="post_training_static_quant"
        extra_cmd='--dataset_config_name=wikitext-2-raw-v1'
    elif [ "${topology}" = "reformer_crime_and_punishment" ]; then
        TASK_NAME='crime_and_punish'
        model_name_or_path=$input_model 
        model_type='reformer'
        SCRIPTS=examples/language-modeling/run_clm_tune.py
        approach="post_training_static_quant"
    elif [ "${topology}" = "xlm-roberta-base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='xlm-roberta'
        approach="post_training_static_quant"
    elif [ "${topology}" = "flaubert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='flaubert'
        approach="post_training_static_quant"
    elif [ "${topology}" = "barthez_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='barthez'
        approach="post_training_static_quant"
    elif [ "${topology}" = "longformer_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='longformer'
        approach="post_training_static_quant"
    elif [ "${topology}" = "layoutlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='layoutlm'
        approach="post_training_static_quant"
    elif [ "${topology}" = "deberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='deberta'
        approach="post_training_static_quant"
    elif [ "${topology}" = "squeezebert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model
        model_type='squeezebert'
        approach="post_training_static_quant"
    elif [ "${topology}" = "transfo_xl_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='transfo-xl-wt103'
    elif [ "${topology}" = "ctrl_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='ctrl'
    elif [ "${topology}" = "xlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=$input_model 
        model_type='xlm-mlm-en-2048'
    fi

    sed -i "/: bert/s|name:.*|name: $model_type|g" conf.yaml
    sed -i "/approach:/s|approach:.*|approach: $approach|g" conf.yaml

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
    elif [ "${SCRIPTS}" = "examples/language-modeling/run_clm_tune.py" ]; then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${TASK_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${input_model} \
            --tune \
            ${extra_cmd}
    fi

}

main "$@"

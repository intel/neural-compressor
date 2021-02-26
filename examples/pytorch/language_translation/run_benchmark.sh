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
    SCRIPTS=examples/run_glue_tune.py
    MAX_SEQ_LENGTH=128
    model_type='bert'

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --iter ${iters} --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='MRPC'
        model_name_or_path='bert-base-uncased'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "distilbert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='distilbert-base-uncased'
        model_type='distilbert'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_base_CoLA" ]; then
        TASK_NAME='CoLA'
        model_name_or_path='bert-base-uncased'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_base_STS-B" ]; then
        TASK_NAME='STS-B'
        model_name_or_path='bert-base-uncased'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_base_SST-2" ]; then
        TASK_NAME='SST-2'
        model_name_or_path='bert-base-uncased'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_base_RTE" ]; then
        TASK_NAME='RTE'
        model_name_or_path='bert-base-uncased'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_large_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_large_SQuAD" ]; then
        SCRIPTS=examples/run_squad_tune.py
        TASK_NAME='SQuAD'
        MAX_SEQ_LENGTH=384
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--doc_stride 128 --do_lower_case'
    elif [ "${topology}" = "bert_large_QNLI" ]; then
        TASK_NAME='QNLI'
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='RTE'
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "bert_large_CoLA" ]; then
        TASK_NAME='CoLA'
        model_name_or_path='bert-large-uncased-whole-word-masking'
        extra_cmd='--do_lower_case'
    elif [ "${topology}" = "xlnet_base_cased_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='xlnet-base-cased'
        model_type='xlnet'
    elif [ "${topology}" = "roberta_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='roberta-base'
        model_type='roberta'
    elif [ "${topology}" = "camembert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path='camembert-base'
        model_type='camembert'
    elif [ "${topology}" = "gpt_WikiText" ];then
        model_name_or_path='openai-gpt'
        model_type='openai-gpt'
        SCRIPTS=examples/run_lm_tune.py
        train_data='wiki.train.raw'
        test_data='wiki.test.raw'
    elif [ "${topology}" = "gpt2_WikiText" ]; then
        model_type='gpt2'
        model_name_or_path='gpt2'
        SCRIPTS=examples/run_lm_tune.py
        train_data='wiki.train.raw'
        test_data='wiki.test.raw'
    elif [ "${topology}" = "ctrl_WikiText" ]; then
        model_type='ctrl'
        model_name_or_path='ctrl'
        SCRIPTS=examples/run_lm_tune.py
        train_data='wiki.train.raw'
        test_data='wiki.test.raw'
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    if [ "${SCRIPTS}" = "examples/run_glue_tune.py" -o "${SCRIPTS}" = "examples/run_squad_tune.py" ];then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_type ${model_type} \
            --model_name_or_path ${model_name_or_path} \
            --task_name ${TASK_NAME} \
            --do_eval \
            --data_dir ${dataset_location} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --per_gpu_eval_batch_size ${batch_size} \
            --no_cuda \
            --output_dir ${input_model} \
            ${mode_cmd} \
            ${extra_cmd}
    elif [ "${SCRIPTS}" = "examples/run_lm_tune.py" ]; then
        python -u $SCRIPTS \
            --tuned_checkpoint ${tuned_checkpoint} \
            --model_type ${model_type} \
            --model_name_or_path ${model_name_or_path} \
            --do_eval \
            --train_data_file ${dataset_location}${train_data} \
            --eval_data_file ${dataset_location}${test_data} \
            --per_gpu_eval_batch_size ${batch_size} \
            --no_cuda \
            --output_dir ${input_model} \
            ${mode_cmd} \
            ${extra_cmd}
    fi

}

main "$@"

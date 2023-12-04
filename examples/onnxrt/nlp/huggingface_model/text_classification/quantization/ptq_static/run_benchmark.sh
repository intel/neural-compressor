#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    
    if [[ "${input_model}" =~ "bert-base-uncased" ]]; then
        model_name_or_path="Intel/bert-base-uncased-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "roberta-base" ]]; then
        model_name_or_path="Intel/roberta-base-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "xlm-roberta-base" ]]; then
        model_name_or_path="Intel/xlm-roberta-base-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "camembert-base" ]]; then
        model_name_or_path="Intel/camembert-base-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "distilbert-base" ]]; then
        model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        TASK_NAME='sst-2'
    fi
    if [[ "${input_model}" =~ "albert-base" ]]; then
        model_name_or_path="Alireza1044/albert-base-v2-sst2"
        TASK_NAME='sst-2'
    fi
    if [[ "${input_model}" =~ "MiniLM-L6" ]]; then
        model_name_or_path="philschmid/MiniLM-L6-H384-uncased-sst2"
        TASK_NAME='sst-2'
    fi
    if [[ "${input_model}" =~ "MiniLM-L12" ]]; then
        model_name_or_path="Intel/MiniLM-L12-H384-uncased-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "bert-base-cased" ]]; then
        model_name_or_path="bert-base-cased-finetuned-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "xlnet-base-cased" ]]; then
        model_name_or_path="Intel/xlnet-base-cased-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "bert-mini" ]]; then
        model_name_or_path="M-FAC/bert-mini-finetuned-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "electra-small-discriminator" ]]; then
        model_name_or_path="Intel/electra-small-discriminator-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "bart" ]]; then
        model_name_or_path="Intel/bart-large-mrpc"
        TASK_NAME='mrpc'
    fi
    if [[ "${input_model}" =~ "deberta" ]]; then
        model_name_or_path="microsoft/deberta-v3-base"
        TASK_NAME='mrpc'
    fi

    python main.py \
            --model_name_or_path ${model_name_or_path} \
            --model_path ${input_model} \
            --data_path ${dataset_location} \
            --task ${TASK_NAME} \
            --mode=${mode} \
            --batch_size=${batch_size} \
            --benchmark
            
}

main "$@"


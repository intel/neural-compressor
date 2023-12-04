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
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
    elif [[ "${input_model}" =~ "bert-base-multilingual" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
    elif [[ "${input_model}" =~ "distilbert-base-uncased" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
    elif [[ "${input_model}" =~ "bert-large-uncased" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
    elif [[ "${input_model}" =~ "roberta-large" ]]; then
        model_name_or_path="deepset/roberta-large-squad2"
        extra_cmd='--version_2_with_negative=True'
    fi

    python main.py \
            --input_model=${input_model} \
            --mode=${mode} \
            --model_name_or_path=${model_name_or_path} \
            --output_dir='./output' \
            --overwrite_output_dir \
            --dataset_name=squad \
            --batch_size=${batch_size} \
            --do_eval \
            --benchmark \
             ${extra_cmd}
            
}

main "$@"
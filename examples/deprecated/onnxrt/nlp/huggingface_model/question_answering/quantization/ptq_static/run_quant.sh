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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --quant_format=*)
          quant_format=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
  
    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
        num_heads=12
        hidden_size=768
    elif [[ "${input_model}" =~ "bert-base-multilingual" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
        num_heads=12
        hidden_size=768
    elif [[ "${input_model}" =~ "distilbert-base-uncased" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        num_heads=12
        hidden_size=768
    elif [[ "${input_model}" =~ "bert-large-uncased" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        num_heads=16
        hidden_size=1024
    elif [[ "${input_model}" =~ "roberta-large" ]]; then
        model_name_or_path="deepset/roberta-large-squad2"
        num_heads=16
        hidden_size=1024
        extra_cmd='--version_2_with_negative=True'
    fi

    python main.py \
            --input_model ${input_model} \
            --dataset_name squad \
            --save_path ${output_model} \
            --output_dir './output' \
            --overwrite_output_dir \
            --model_name_or_path=${model_name_or_path} \
            --num_heads ${num_heads} \
            --hidden_size ${hidden_size} \
            --do_eval \
            --tune \
            ${extra_cmd}
}

main "$@"
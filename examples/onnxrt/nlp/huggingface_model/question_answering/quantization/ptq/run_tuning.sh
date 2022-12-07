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
    esac
  done

}

# run_tuning
function run_tuning {
  
    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
        num_heads=12
        hidden_size=768
    elif [[ "${input_model}" =~ "bert-base" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
        num_heads=12
        hidden_size=768
    fi

    python main.py \
            --model_path ${input_model} \
            --save_path ${output_model} \
            --output_dir './output' \
            --model_name_or_path=${model_name_or_path} \
            --num_heads ${num_heads} \
            --hidden_size ${hidden_size} \
            --tune 
}

main "$@"
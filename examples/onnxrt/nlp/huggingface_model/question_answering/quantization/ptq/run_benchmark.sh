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
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
    elif [[ "${input_model}" =~ "bert-base" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
    fi

    python main.py \
            --model_path ${input_model} \
            --config ${config} \
            --mode=${mode} \
            --model_name_or_path=${model_name_or_path} \
            --output_dir './output' \
            --benchmark
            
}

main "$@"
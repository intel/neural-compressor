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
    
    python main.py \
           --input_model ${input_model} \
           --model_name_or_path HYPJUDY/layoutlmv3-base-finetuned-funsd \
           --dataset_name funsd \
           --save_path ${output_model} \
           --output_dir ./output_dir \
           --overwrite_output_dir \
           --quant_format ${quant_format} \
           --tune
}

main "$@"
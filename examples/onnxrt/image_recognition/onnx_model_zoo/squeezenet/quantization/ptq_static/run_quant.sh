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
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --label_path=*)
          label_path=$(echo $var |cut -f2 -d=)
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
            --model_path ${input_model} \
            --dataset_location ${dataset_location} \
            --label_path ${label_path-${dataset_location}/../val.txt} \
            --output_model ${output_model} \
            --quant_format ${quant_format-default} \
            --tune
}

main "$@"

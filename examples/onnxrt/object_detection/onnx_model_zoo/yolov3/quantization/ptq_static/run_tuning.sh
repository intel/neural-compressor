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
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --label_path=*)
          label_path=$(echo $var |cut -f2 -d=)
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
    if [ ! $label_path ]; then
        label_path='label_map.yaml'
    fi

    python main.py \
            --model_path ${input_model} \
            --output_model ${output_model} \
            --data_path ${dataset_location} \
            --label_path ${label_path} \
            --quant_format ${quant_format} \
            --tune
}

main "$@"
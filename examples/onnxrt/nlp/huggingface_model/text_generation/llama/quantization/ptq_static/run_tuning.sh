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
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --alpha=*)
          alpha=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    python main.py \
            --quant_format ${quant_format-QOperator} \
            --model_path ${input_model} \
            --output_model ${output_model} \
            --batch_size ${batch_size-1} \
            --alpha ${alpha-0.5} \
            --dataset ${dataset-NeelNanda/pile-10k} \
            --tune
}

main "$@"


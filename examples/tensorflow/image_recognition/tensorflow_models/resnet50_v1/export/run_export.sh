#!/bin/bash
set -x

function main {
  init_params "$@"
  run_export

}

# init params
function init_params {
  quant_format=qdq
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
      --dtype=*)
          dtype=$(echo $var |cut -f2 -d=)
      ;;
      --quant_format=*)
          quant_format=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_export
function run_export {
    if [ ${dtype} == 'int8' ]
    then
        python main.py \
                --input-graph ${input_model} \
                --output-graph ${output_model} \
                --dtype ${dtype} \
                --quant_format ${quant_format} \
                --dataset_location ${dataset_location} \
                --export
    else
        python main.py \
                --input-graph ${input_model} \
                --output-graph ${output_model} \
                --dtype ${dtype} \
                --quant_format ${quant_format} \
                --export
    fi
}

main "$@"

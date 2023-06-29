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
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --label_path=*)
          label_path=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

function run_benchmark {
    cd table-transformer/src
    python main.py \
            --input_onnx_model ${input_model} \
            --data_root_dir ${dataset_location} \
            --table_words_dir ${label_path} \
            --mode ${mode} \
            --data_type structure \
            --device cpu \
            --config_file structure_config.json \
}

main "$@"

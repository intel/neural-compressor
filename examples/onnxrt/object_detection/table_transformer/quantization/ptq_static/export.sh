#!/bin/bash
set -x

function main {
  init_params "$@"
  export_model

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
    esac
  done

}

function export_model {
    cd table-transformer/src
    python main.py \
            --model_load_path ${input_model} \
            --output_model ${output_model} \
            --data_root_dir ${dataset_location}/PubTables1M-Structure-PASCAL-VOC \
            --table_words_dir ${dataset_location}/PubTables1M-Table-Words-JSON \
            --mode 'export' \
            --data_type structure \
            --device cpu \
            --config_file structure_config.json
}

main "$@"

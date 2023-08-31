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
    esac
  done

}

function run_tuning {

    # Check if the directory exists
    if [ ! -d "./table-transformer" ]; then
        # If the directory doesn't exist, create it
	      bash prepare.sh
    fi

    cd table-transformer/src
    python main.py \
            --input_onnx_model ${input_model} \
            --output_model ${output_model} \
            --data_root_dir ${dataset_location} \
            --table_words_dir ${dataset_location}/words \
            --mode quantize \
            --data_type structure \
            --device cpu \
            --config_file structure_config.json
}

main "$@"

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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

function run_benchmark {

    # Check if the directory exists
    if [ ! -d "./table-transformer" ]; then
        # If the directory doesn't exist, create it
	      bash prepare.sh
    fi

    if [[ "${input_model}" =~ "structure" ]]; then
        task_data_dir="PubTables-1M-Structure"
        data_type="structure"
        config_file="structure_config.json"
    fi
    if [[ "${input_model}" =~ "detection" ]]; then
        task_data_dir="PubTables-1M-Detection"
        data_type="detection"
        config_file="detection_config.json"
    fi

    input_model=$(realpath "$input_model")

    cd table-transformer/src
    python main.py \
            --input_onnx_model ${input_model} \
            --data_root_dir "${dataset_location}/${task_data_dir}" \
            --table_words_dir "${dataset_location}/${task_data_dir}/words" \
            --mode ${mode} \
            --data_type ${data_type} \
            --device cpu \
            --config_file ${config_file}
}

main "$@"

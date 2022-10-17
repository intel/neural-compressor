#!/bin/bash
set -x

function main {

  init_params "$@"
  export BUILD_DIR=${dataset_location}
  export nnUNet_preprocessed=${BUILD_DIR}/preprocessed_data
  export nnUNet_raw_data_base=${BUILD_DIR}/raw_data
  export RESULTS_FOLDER=${BUILD_DIR}/result
  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    config=$topology'.yaml'
    python run_accuracy.py \
      --input-model=${input_model} \
      --output-model=${output_model} \
      --data-location=${dataset_location} \
      --calib-preprocess=${BUILD_DIR}/calib_preprocess \
      --config=${config} \
      --mode=tune 
}

main "$@"

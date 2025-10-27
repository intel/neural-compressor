#!/bin/bash
set -x

function main {

  init_params "$@"
  export BUILD_DIR=${dataset_location}
  export nnUNet_preprocessed=${BUILD_DIR}/preprocessed_data
  export nnUNet_raw_data_base=${BUILD_DIR}/raw_data
  export RESULTS_FOLDER=${BUILD_DIR}/result
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=1
  for var in "$@"
  do
    case $var in
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${bfloat16} == "true" ]]; then
        extra_cmd="--bfloat16"
    else
        extra_cmd=""
    fi

    python run_accuracy.py \
      --input-model=${input_model} \
      --data-location=${dataset_location} \
      --calib-preprocess=${BUILD_DIR}/calib_preprocess \
      --batch-size=${batch_size} \
      --mode=${mode} \
      ${extra_cmd}
}

main "$@"

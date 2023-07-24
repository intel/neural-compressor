#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
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
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
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
    extra_cmd=""
    if [ -n "$topology" ];then
        extra_cmd=$extra_cmd"--profile ${topology}-pytorch "
    fi
    if [ -n "$dataset_location" ];then
        if [ "${dataset_location}" != "convert_dataset" ]; then
            ln -s ${dataset_location}"/preprocessed" "preprocessed"
            ln -s ${dataset_location} "convert_dataset"
        fi
        extra_cmd=$extra_cmd"--dataset-path ${dataset_location} --cache_dir ${dataset_location} "
    fi
    if [ -n "$input_model" ];then
        extra_cmd=$extra_cmd"--model ${input_model} "
    fi
    if [ -n "$tuned_checkpoint" ];then
        extra_cmd=$extra_cmd"--tuned_checkpoint ${tuned_checkpoint} "
    fi

    OUTPUT_DIR=`pwd`/output/pytorch-cpu/${topology}
    if [ ! -d $OUTPUT_DIR ]; then
        mkdir -p $OUTPUT_DIR
    fi

    python python/main.py --tune \
                          --accuracy \
                          --backend pytorch-native \
                          --output ${OUTPUT_DIR} \
                          ${extra_cmd}
}

main "$@"


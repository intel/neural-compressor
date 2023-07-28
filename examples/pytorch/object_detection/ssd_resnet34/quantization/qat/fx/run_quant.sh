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
    if [ -n "$dataset_location" ];then
        extra_cmd=$extra_cmd"--data ${dataset_location} "
    fi
    if [ -n "$input_model" ];then
        extra_cmd=$extra_cmd"--checkpoint ${input_model} "
    fi
    if [ -n "$tuned_checkpoint" ];then
        extra_cmd=$extra_cmd"--tuned_checkpoint ${tuned_checkpoint} "
    fi

    python ssd/main.py --epochs 1 \
                       --tune \
                       ${extra_cmd}
}

main "$@"

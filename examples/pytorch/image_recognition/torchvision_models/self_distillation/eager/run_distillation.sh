#!/bin/bash
set -x

function main {
  init_params "$@"
  run_distillation

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --use_cpu=*)
          use_cpu=$(echo $var |cut -f2 -d=)
      ;;
      --hpo=*)
          hpo=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_distillation {
    extra_cmd=${dataset_location}
    if [ 1 = "${use_cpu}" ];then
        extra_cmd=${extra_cmd}" --cpu"
    fi
    if [ 1 = "${hpo}" ];then
        extra_cmd=${extra_cmd}" --hpo"
    fi
    python main.py \
           --topology=${topology} \
           --output-model=${output_model} \
           ${extra_cmd}
}

main "$@"

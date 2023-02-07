#!/bin/bash
set -x

function main {
  init_params "$@"
  download_data

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --data_dir=*)
          data_dir=$(echo $var |cut -f2 -d=)
      ;;
      --task_name=*)
          task_name=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function download_data {
    wget https://raw.githubusercontent.com/huggingface/transformers/f98ef14d161d7bcdc9808b5ec399981481411cc1/utils/download_glue_data.py
    python download_glue_data.py --data_dir=${data_dir} --tasks=${task_name}
}

main "$@"


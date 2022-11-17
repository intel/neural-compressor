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
      --input_dir=*)
          input_dir=$(echo $var |cut -f2 -d=)
      ;;
      --task_name=*)
          task_name=$(echo $var |cut -f2 -d=)
      ;;
      --max_len=*)
          max_len=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function export_model {
    python export.py --input_dir ${input_dir} --task_name ${task_name} --output_model ${output_model} 
}

main "$@"


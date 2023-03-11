#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  int8=false
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --int8)
          int8=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {

  if [ ${int8} == true ]; then
     python benchmark.py \
        --model_name_or_path ${input_model} \
        --int8
  else
     python benchmark.py \
        --model_name_or_path ${input_model}
  fi

}

main "$@"

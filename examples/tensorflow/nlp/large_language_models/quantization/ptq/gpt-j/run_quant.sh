#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  fp32_path="./gpt-j-6B"
  for var in "$@"
  do
    case $var in
      --fp32_path=*)
          fp32_path=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    python main.py \
        --fp32_path=${fp32_path} \
        --output_model=${output_model} \
        --output_dir="./outputs" \
        --tune
}

main "$@"

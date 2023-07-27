#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
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
      --input_graph=${input_model} \
      --inputs_file=${dataset_location}/newstest2014.en \
      --reference_file=${dataset_location}/newstest2014.de \
      --vocab_file=${dataset_location}/vocab.txt \
      --output_model=${output_model} \
      --tune
}

main "$@"


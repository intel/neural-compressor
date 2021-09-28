#!/bin/bash
set -x

export GLOG_minloglevel=2

#default batch_size
batch_size=1

function main {

  init_params "$@"

  run_tuning

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    python run_engine.py \
      --input_model=${input_model} \
      --output_model=$output_model \
      --data_dir=${dataset_location}/dev-v1.1.json \
      --vocab_file=${dataset_location}/vocab.txt \
      --config=$config \
      --tune \

}

main "$@"

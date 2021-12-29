#!/bin/bash
# set -x

export GLOG_minloglevel=2

#default batch_size
batch_size=1
tokenizer_dir=bhadresh-savani/distilbert-base-uncased-emotion

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
      --tokenizer_dir=*)
          tokenizer_dir=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    python run_engine.py \
      --input_model=${input_model} \
      --output_model=$output_model \
      --data_dir=${dataset_location} \
      --tokenizer_dir=$tokenizer_dir \
      --config=$config \
      --tune \

}

main "$@"

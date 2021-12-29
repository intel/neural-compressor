#!/bin/bash
# set -x

export GLOG_minloglevel=2

batch_size=1
tokenizer_dir=distilbert-base-uncased-finetuned-sst-2-english

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --tokenizer_dir=*)
          tokenizer_dir=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {
    python run_engine.py \
      --input_model=${input_model} \
      --data_dir=${dataset_location}/ \
      --tokenizer_dir=$tokenizer_dir \
      --batch_size=${batch_size} \
      --config=$config \
      --benchmark \
      --mode=$mode \

}

main "$@"

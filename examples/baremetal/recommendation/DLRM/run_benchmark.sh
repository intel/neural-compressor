#!/bin/bash
# set -x

export GLOG_minloglevel=2

batch_size=32
output_model=./ir
dataset=kaggle
mlperf_bin_loader=false


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
      --dataset=*)
          dataset=$(echo "$var" |cut -f2 -d=)
      ;;
      --mlperf_bin_loader*)
          mlperf_bin_loader=$(echo "$var")
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {
    if [ ${mlperf_bin_loader} == '--mlperf_bin_loader' ]
    then
        python run_engine.py \
          --input_model=${input_model} \
          --output_model=$output_model \
          --raw_path=${dataset_location} \
          --pro_data=${dataset_location} \
          --batch_size=${batch_size} \
          --config=$config \
          --dataset=$dataset \
          --benchmark \
          --mlperf_bin_loader \
          --mode=$mode
    else
        python run_engine.py \
          --input_model=${input_model} \
          --output_model=$output_model \
          --raw_path=${dataset_location}/train.txt \
          --pro_data=${dataset_location}/kaggleAdDisplayChallenge_processed.npz \
          --batch_size=${batch_size} \
          --config=$config \
          --dataset=$dataset \
          --benchmark \
          --mode=$mode \
          --mlperf_bin_loader
    fi
}

main "$@"

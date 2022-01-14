#!/bin/bash
# set -x

export GLOG_minloglevel=2

#default batch_size
batch_size=32
output_model=./ir
dataset=kaggle
mlperf_bin_loader=False

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
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
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

function run_tuning {
    if [ ${mlperf_bin_loader} == '--mlperf_bin_loader' ]
    then
        python run_engine.py \
          --input_model=${input_model} \
          --output_model=$output_model \
          --raw_path=${dataset_location} \
          --pro_data=${dataset_location} \
          --config=$config \
          --batch_size=${batch_size} \
          --dataset=$dataset \
          --tune \
          --mlperf_bin_loader
    else
        python run_engine.py \
          --input_model=${input_model} \
          --output_model=$output_model \
          --raw_path=${dataset_location}/train.txt \
          --pro_data=${dataset_location}/kaggleAdDisplayChallenge_processed.npz \
          --batch_size=${batch_size} \
          --config=$config \
          --dataset=$dataset \
          --tune
    fi


}

main "$@"

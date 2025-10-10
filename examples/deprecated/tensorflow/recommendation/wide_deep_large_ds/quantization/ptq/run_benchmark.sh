#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
  run_benchmark

}

# init params
function init_params {
  iters=1000
  for var in "$@"
  do
    case $var in
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
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
    esac
  done

}

function define_mode {
    if [[ ${mode} == "accuracy" ]]; then
      mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
      mode_cmd=" --performance"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_tuning
function run_benchmark {
    #numactl -N 0 -m 0 \
    python inference.py \
            --input_graph ${input_model} \
            --evaluation_data_location ${dataset_location}/eval_processed_data.tfrecords \
            --batch_size ${batch_size} \
            --num_inter_threads 4 \
            ${mode_cmd}
}

main "$@"

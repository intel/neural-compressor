#!/bin/bash
# set -x

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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

function define_mode {
    
    if [[ ${mode} == "accuracy" ]]; then
        mode="accuracy"
    elif [[ ${mode} == "benchmark" ]]; then
        mode="benchmark"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
}
        
# run_benchmark
function run_benchmark {
    config=$topology'.yaml'
    python main.py \
      --input_graph=${input_model} \
      --inputs_file=${dataset_location}/newstest2014.en \
      --reference_file=${dataset_location}/newstest2014.de \
      --vocab_file=${dataset_location}/vocab.txt \
      --config=${config} \
      --mode=${mode} \
      --iters=${iters} \
      --batch_size=${batch_size} 
}

main "$@"


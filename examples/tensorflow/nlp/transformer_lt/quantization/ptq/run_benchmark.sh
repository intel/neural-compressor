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
    esac
  done

}

function define_mode {
    
    if [[ ${mode} == "accuracy" ]]; then
        mode="accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode="performance"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
}
        
# run_benchmark
function run_benchmark {
    python main.py \
      --input_graph=${input_model} \
      --inputs_file=${dataset_location}/newstest2014.en \
      --reference_file=${dataset_location}/newstest2014.de \
      --vocab_file=${dataset_location}/vocab.txt \
      --benchmark \
      --mode=${mode} \
      --iters=${iters} \
      --batch_size=${batch_size} 
}

main "$@"


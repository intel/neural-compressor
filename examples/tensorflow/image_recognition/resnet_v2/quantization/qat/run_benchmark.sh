#!/bin/bash
set -x

function main {
  batch_size=32
  input_model="./baseline_model"
  iters=100
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)    
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)    
      ;;
      --iters=*)
          iters=$(echo $var |cut -f2 -d=)    
      ;;
    esac
  done

  if [ ! -n "${input_model}" ] ;then
    input_model="./baseline_model"
  fi
}

# run_tuning
function run_benchmark {

    python main.py \
            --input_model=${input_model} \
            --mode=${mode} \
            --batch_size=${batch_size} \
            --iters=${iters} \
            --benchmark
}

main "$@"

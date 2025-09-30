#!/bin/bash
set -x

function main {

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
    esac
  done

}

# run_benchmark
function run_benchmark {
    
    python main.py \
           --input_model ${input_model} \
           --model_name_or_path HYPJUDY/layoutlmv3-base-finetuned-funsd \
           --dataset_name funsd \
           --output_dir ./output_dir \
           --overwrite_output_dir \
           --batch_size=${batch_size} \
           --mode=${mode} \
           --benchmark \
            
}

main "$@"
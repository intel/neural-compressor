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
      --tasks=*)
          tasks=$(echo $var |cut -f2 -d=)
      ;;
      --intra_op_num_threads=*)
          intra_op_num_threads=$(echo $var |cut -f2 -d=)
      ;;
      --tokenizer=*)
          tokenizer=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    
    # Check if the input_model ends with the filename extension ".onnx"
    if [[ $input_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        input_model=$(dirname "$input_model")
    fi

    python main.py \
            --model_path ${input_model} \
            --mode=${mode} \
            --batch_size=${batch_size-1} \
            --tokenizer=${tokenizer-${input_model}} \
            --tasks=${tasks-lambada_openai} \
            --intra_op_num_threads=${intra_op_num_threads-4} \
            --benchmark
            
}

main "$@"


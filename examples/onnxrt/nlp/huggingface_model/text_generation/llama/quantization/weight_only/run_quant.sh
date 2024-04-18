#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --tokenizer=*)
          tokenizer=$(echo $var |cut -f2 -d=)
      ;;
      --algorithm=*)
          algorithm=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    # Check if the input_model ends with the filename extension ".onnx"
    if [[ $input_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        input_model=$(dirname "$input_model")
    fi

    # Check if the output_model ends with the filename extension ".onnx"
    if [[ $output_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        output_model=$(dirname "$output_model")
    fi

    # Check if the directory exists
    if [ ! -d "$output_model" ]; then
        # If the directory doesn't exist, create it
	mkdir -p "$output_model"
	echo "Created directory $output_model"
    fi

    python main.py \
            --model_path ${input_model} \
	        --tokenizer ${tokenizer-meta-llama/Llama-2-7b-hf} \
            --output_model ${output_model} \
            --batch_size ${batch_size-1} \
            --dataset ${dataset-NeelNanda/pile-10k} \
	        --algorithm ${algorithm-WOQ_TUNE} \
	        --tasks ${tasks-lambada_openai} \
            --tune
}

main "$@"


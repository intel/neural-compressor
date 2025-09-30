#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
  run_benchmark

}

# init params
function init_params {
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
      mode_cmd=""
    elif [[ ${mode} == "performance" ]]; then
      mode_cmd=" --iter ${iters}"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_benchmark
function run_benchmark {
    if [[ "${input_model}" =~ "gpt2" ]]; then
        model_name_or_path="gpt2"
    fi
    if [[ "${input_model}" =~ "distilgpt2" ]]; then
        model_name_or_path="distilgpt2"
    fi

    python main.py --model_path ${input_model} \
                        --data_path ${dataset_location} \
                        --model_name_or_path ${model_name_or_path} \
                        --eval_batch_size ${batch_size} \
                        --benchmark \
                        --mode ${mode} \
                        ${mode_cmd}
}

main "$@"

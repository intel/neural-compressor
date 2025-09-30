#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=32
  tuned_checkpoint=saved_results
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
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
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${input_model: -5:5} == ".onnx" ]]; then
        python onnx_evaluation.py \
                --model_path ${input_model} \
                --dataset_location ${dataset_location} \
                --batch_size=${batch_size} \
                --iters=${iters} \
                --mode=${mode} \
                --benchmark
    else
        if [[ ${mode} == "accuracy" ]]; then
            mode_cmd=" --accuracy"
        elif [[ ${mode} == "performance" ]]; then
            mode_cmd=" --iter ${iters} --performance "
        else
            echo "Error: No such mode: ${mode}"
            exit 1
        fi

        extra_cmd=""
        if [[ ${int8} == "true" ]]; then
            extra_cmd=$extra_cmd" --int8"
        fi

        python main.py \
                --pretrained \
                --tuned_checkpoint ${tuned_checkpoint} \
                -b ${batch_size} \
                -a ${input_model} \
                ${mode_cmd} \
                ${extra_cmd} \
                ${dataset_location}
    fi
}

main "$@"

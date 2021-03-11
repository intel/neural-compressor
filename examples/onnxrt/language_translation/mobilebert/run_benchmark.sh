#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
  run_benchmark

}

# init params
function init_params {
  bert_yaml="./mobilebert.yaml"
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
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
      mode_cmd=" --benchmark --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --benchmark_nums ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_benchmark
function run_benchmark {
    if [ "${topology}" = "mobilebert_MRPC" ];then
      task_name='mrpc'
      model_name_or_path='google/mobilebert-uncased'
    fi
    python bert_base.py --model_path ${input_model} \
                        --data_dir ${dataset_location} \
                        --task_name ${task_name} \
                        --input_dir ${model_name_or_path} \
                        --config ${bert_yaml} \
                        --eval_batch_size ${batch_size} \
                        ${mode_cmd}
}

main "$@"


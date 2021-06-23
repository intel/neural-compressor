#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
  run_benchmark

}

# init params
function init_params {
  gpt2_yaml="./gpt2.yaml"
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
      mode_cmd=" --benchmark --mode=accuracy"
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --iter ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_benchmark
function run_benchmark {
    if [ "${topology}" = "gpt2_lm_wikitext2" ];then
      model_type='gpt2'
      model_name_or_path='gpt2'
      test_data='wiki.test.raw'
    fi
    python gpt2.py --model_path ${input_model} \
                        --eval_data_file ${dataset_location}${test_data} \
                        --model_type ${model_type} \
                        --model_name_or_path ${model_name_or_path} \
                        --config ${gpt2_yaml} \
                        --per_gpu_eval_batch_size ${batch_size} \
                        ${mode_cmd}
}

main "$@"


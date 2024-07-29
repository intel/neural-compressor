#!/bin/bash
set -x

function main {

  export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  tuned_checkpoint=saved_results
  batch_size=20
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
      --optimized=*)
          optimized=$(echo ${var} |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iter ${iters} --performance "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${optimized} == "true" ]]; then
        extra_cmd="--optimized ${dataset_location}"
    else
        extra_cmd="${dataset_location}"
    fi
    if [[ ${mode} == "accuracy" ]]; then
        python main.py \
                --pretrained \
                --tuned_checkpoint ${tuned_checkpoint} \
                -b ${batch_size} \
                -a ${input_model} \
                ${mode_cmd} \
                ${extra_cmd}
    elif [[ ${mode} == "performance" ]]; then
        incbench --num_c 4 main.py \
                --pretrained \
                --tuned_checkpoint ${tuned_checkpoint} \
                -b ${batch_size} \
                -a ${input_model} \
                ${mode_cmd} \
                ${extra_cmd}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
}

main "$@"

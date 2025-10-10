#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  tuned_checkpoint=saved_results
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
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
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
        mode_cmd=" --accuracy_only "
        default_bs=16384
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" -i ${iters} --benchmark "
        default_bs=16
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${batch_size} == '' ]]; then
        batch_size=$default_bs
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd="--int8"
    else
        extra_cmd=""
    fi

    python -u dlrm_s_pytorch_tune.py \
        --tuned_checkpoint ${tuned_checkpoint} \
        --arch-sparse-feature-size=128 \
        --arch-mlp-bot="13-512-256-128" \
        --arch-mlp-top="1024-1024-512-256-1" \
        --max-ind-range=40000000 \
        --data-generation=dataset \
        --data-set=terabyte \
        --raw-data-file=${dataset_location}/day \
        --processed-data-file=${dataset_location}/terabyte_processed.npz \
        --loss-function=bce \
        --round-targets=True \
        --learning-rate=1.0 \
        --mini-batch-size=2048 \
        --print-freq=2048 \
        --print-time \
        --test-freq=102400 \
        --test-mini-batch-size=${batch_size} \
        --test-num-workers=16 \
        --memory-map \
        --mlperf-logging \
        --mlperf-auc-threshold=0.8025 \
        --mlperf-bin-loader \
        --mlperf-bin-shuffle \
        --load-model=${input_model} \
        --inference-only \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"

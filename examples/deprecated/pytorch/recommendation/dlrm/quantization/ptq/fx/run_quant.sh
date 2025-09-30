#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
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
      --output_model=*)
         tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=""

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
            --test-mini-batch-size=16384 \
            --test-num-workers=16 \
            --memory-map \
            --mlperf-logging \
            --mlperf-auc-threshold=0.8025 \
            --mlperf-bin-loader \
            --mlperf-bin-shuffle \
            --load-model=${input_model} \
            --tune \
            --inference-only \
            ${extra_cmd}

}

main "$@"

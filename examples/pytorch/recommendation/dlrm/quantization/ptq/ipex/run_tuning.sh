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

CORES=`lscpu | grep Core | awk '{print $4}'`
# use first socket
numa_cmd="numactl -C 0-$((CORES-1))  "
echo "will run on core 0-$((CORES-1)) on socket 0" 

export OMP_NUM_THREADS=$CORES

# run_tuning
function run_tuning {
  MODEL_SCRIPT=dlrm_s_pytorch.py

  # Create the output directory in case it doesn't already exist
  mkdir -p ${tuned_checkpoint}/dlrm_inference_accuracy_log

  LOG=${tuned_checkpoint}/dlrm_inference_accuracy_log
  CORES=`lscpu | grep Core | awk '{print $4}'`
  ARGS=""

  $numa_cmd python -u $MODEL_SCRIPT \
  --raw-data-file=${dataset_location}/day --processed-data-file=${dataset_location}/terabyte_processed.npz \
  --data-set=terabyte \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=40000000 \
  --numpy-rand-seed=727  --inference-only --ipex-interaction \
  --print-freq=100 --print-time --mini-batch-size=2048 --test-mini-batch-size=16384 \
  --test-freq=2048 --print-auc --tune --save-model=${tuned_checkpoint} $ARGS \
  --load-model=${input_model} --num-cpu-cores=${CORES} | tee $LOG
}

main "$@"

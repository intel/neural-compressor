#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  batch_size=16384
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

# run_tuning
function run_tuning {
    MODEL_SCRIPT=dlrm_s_pytorch.py

    # Create the output directory in case it doesn't already exist
    mkdir -p ${tuned_checkpoint}/dlrm_inference_accuracy_log

    LOG=${tuned_checkpoint}/dlrm_inference_accuracy_log

    CORES=`lscpu | grep Core | awk '{print $4}'`

    ARGS=""
    if [[ ${int8} == "true" ]]; then
        echo "running int8 path"
        ARGS="$ARGS --int8 --int8-configure=${tuned_checkpoint}/best_configure.json"
    else
        echo "running fp32 path"
    fi

    if [[ ${mode} == "accuracy" ]]; then
        numa_cmd="numactl -C 0-$((CORES-1))  "
        echo "will run on core 0-$((CORES-1)) on socket 0" 
        export OMP_NUM_THREADS=$CORES
        $numa_cmd python -u $MODEL_SCRIPT \
        --raw-data-file=${dataset_location}/day --processed-data-file=${dataset_location}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 \
        --numpy-rand-seed=727  --inference-only --ipex-interaction \
        --print-freq=100 --print-time --mini-batch-size=2048 --test-mini-batch-size=16384 \
        --test-freq=2048 --print-auc $ARGS \
        --load-model=${input_model}
    elif [[ ${mode} == "benchmark" ]]; then
        LOG_0="${LOG}/throughput.log"
        export OMP_NUM_THREADS=1
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --enable_jemalloc $MODEL_SCRIPT \
        --raw-data-file=${dataset_location}/day --processed-data-file=${dataset_location}/terabyte_processed.npz \
        --data-set=terabyte \
        --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
        --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
        --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction \
        --numpy-rand-seed=727  --inference-only --num-batches=1000 \
        --print-freq=10 --print-time --mini-batch-size=128 --test-mini-batch-size=${batch_size} \
        --share-weight-instance=$CORES --num-cpu-cores=$CORES\
        $ARGS |tee $LOG_0
        wait
        set +x

        throughput=$(grep 'throughput:' ${LOG}/throughput.log |sed -e 's/.*throughput//;s/[^0-9.]//g' |awk '
        BEGIN {
                sum = 0;
                i = 0;
              }
              {
                sum = sum + $1;
                i++;
              }
        END   {
        sum = sum / i;
                printf("%.3f", sum);
        }')
        echo ""dlrm";"batch_size";${batch_size};"Throughput": ${throughput}"

    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
}

main "$@"

#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="whisper"
export MODEL="whisper"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance/nvidia
export MAX_LATENCY=10000000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export SCENARIO=${SCENARIO}
  export MODE=${MODE}

  # Standard ENV settings (potentially redundant)
  export MODEL_DIR=${MODEL_DIR}/whisper-large-v3
  export DATA_DIR=${DATA_DIR}
  export MANIFEST_FILE=${DATA_DIR}/dev-all-repack.json
  export USER_CONF=${USER_CONF}
  export RUN_LOGS=${RUN_LOGS}


  export VLLM_USE_V1=1
  # export VLLM_LOGGING_LEVEL="DEBUG"
  # export VLLM_ENABLE_V1_MULTIPROCESSING=0
  # export VLLM_ATTENTION_BACKEND="FLASH_ATTN_VLLM_V1"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export ONEAPI_DEVICE_SELECTOR=level_zero:0
#  export EXTRA_ARGS="--accuracy"
  if [ "${MODE}" == "Compliance" ]; then
    export MODE="Performance"
  fi

  export EXTRA_ARGS=""
  if [ "${MODE}" == "Accuracy" ]; then
      export EXTRA_ARGS="--accuracy"
  fi
  python main.py \
      --dataset_dir ${DATA_DIR} \
      --model_path ${MODEL_DIR} \
      --manifest ${MANIFEST_FILE} \
      --scenario Offline \
      --log_dir ${RUN_LOGS} \
      --num_workers 1 \
      ${EXTRA_ARGS}


#  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD
#  export HF_HOME=${DATA_DIR}/huggingface
#
#  # Core count adaptation
#  export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
#  export NUM_CACHE_NODES=$(find /sys/devices/system/cpu/cpu*/cache -name level 2>/dev/null | xargs grep -l '^3$' 2>/dev/null | sed 's|/level|/shared_cpu_map|' | xargs -r cat 2>/dev/null | sort -u | wc -l)
#  # With clustered L3 cache, use number of L3 caches instead of number of NUMA nodes
#  if ((NUM_CACHE_NODES > NUM_NUMA_NODES)); then
#    export NUM_NODES=NUM_CACHE_NODES
#  else
#    export NUM_NODES=NUM_NUMA_NODES
#  fi
#
#  export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
#  export CORES_PER_NODE=$(($NUM_CORES / $NUM_NODES))
#
#  # Golden config is 6 cores/inst, bs=96/inst
#  # Allow degraded setup going from there
#  MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $7}')
#  MEM_PER_NODE=$(($MEM_AVAILABLE / $NUM_NODES))
#  MEM_MODEL=4
#  
#  # Use BATCH_SIZE=96 if memory allows, 64, 48, 32, and lower if not able to fit
#  # Aim for 4<=CORES_PER_INST<=10
#  # Loop through the range [4, 10]
#  MAX_CORES_PER_NODE=0
#  for i in {4..10}; do
#    # Check for divisibility
#    MAX_INSTANCES_PER_NODE=$(( CORES_PER_NODE / i ))
#    TEST_CORES_PER_NODE=$(( MAX_INSTANCES_PER_NODE * i ))
#    if (( TEST_CORES_PER_NODE > MAX_CORES_PER_NODE )); then
#      CORES_PER_INST_MIN=$i
#      MAX_CORES_PER_NODE=${TEST_CORES_PER_NODE}
#    fi
#  done
#
#  # 1500 + 448 max tokens, 1280 d_model, 32 layers, k+v, 1 bit/value 8-bit
#  CACHE_FACTOR_PER_BATCH=$((1948 * 32 * 1280 * 2))
#  GIB_DIVISOR=1073741824
#
#  # Iterate through potential batch sizes from largest to smallest
#  for BATCH_SIZE in 96 64 48 32; do
#    MEM_CACHE=$((BATCH_SIZE * CACHE_FACTOR_PER_BATCH / GIB_DIVISOR))
#    MEM_TOTAL=$((MEM_MODEL + MEM_CACHE))
#
#    # Ensure MEM_TOTAL and INSTS_PER_NODE are not zero to prevent division errors
#    if (( MEM_TOTAL > 0 && (INSTS_PER_NODE = MEM_PER_NODE / MEM_TOTAL) > 0 )); then
#      CORES_PER_INST=$((CORES_PER_NODE / INSTS_PER_NODE))
#    else
#      # Set to a high value if instance cannot run, to try the next smaller batch size
#      CORES_PER_INST=999
#    fi
#
#    # If cores per instance is acceptable, we found our batch size and can exit the loop
#    if ((CORES_PER_INST <= 10)); then
#      break
#    fi
#  done
#
#  if ((CORES_PER_INST < CORES_PER_INST_MIN)); then
#    CORES_PER_INST=$CORES_PER_INST_MIN;
#  fi
#
#  # Find the smallest CORES_PER_INST so that CORES_PER_NODE is divisible by CORES_PER_INST
#  for ((i=CORES_PER_INST; i<=10; i++)); do
#    if ((CORES_PER_NODE % i == 0)); then
#      CORES_PER_INST=$i
#      break
#    fi
#  done  
#
#  if [ "${SCENARIO}" == "Offline" ]; then
#    export CORES_PER_INST=$CORES_PER_INST
#    export VLLM_CPU_KVCACHE_SPACE=$MEM_CACHE
#  fi
#
#  # Workload run-specific settings
#  export MANIFEST_FILE="${DATA_DIR}/dev-all-repack.json"
#
#  if [ "${MODE}" == "Compliance" ]; then
#    export MODE="Performance"
#  fi
#
#  echo $CORES_PER_INST
#  echo $VLLM_CPU_KVCACHE_SPACE
#  
#  # Using NUMA nodes here to not confuse SUT
#  export INSTS_PER_NODE=$(($NUM_CORES / $NUM_NUMA_NODES / CORES_PER_INST))
#  export NUM_INSTS=$((${INSTS_PER_NODE} * ${NUM_NUMA_NODES}))
#
#  export EXTRA_ARGS=""
#  if [ "${MODE}" == "Accuracy" ]; then
#      export EXTRA_ARGS="--accuracy"
#  fi
#
#  cd ${WORKSPACE_DIR}
#
#  export MODEL_PATH=${MODEL_DIR}/whisper-large-v3-rtn
#
#  # Following v5.0 Inference release, one file handles all scenarios/modes: 
#  python run.py \
#      --dataset_dir ${DATA_DIR} \
#      --model_path ${MODEL_PATH} \
#      --manifest ${MANIFEST_FILE} \
#      --scenario ${SCENARIO} \
#      --log_dir ${RUN_LOGS} \
#      --num_workers $NUM_INSTS \
#      ${EXTRA_ARGS}
}

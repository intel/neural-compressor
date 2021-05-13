CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=8
LOG="./logs"
GLUE_DIR=~/glue_data

if [ ! -d $LOG ];then
   mkdir $LOG
fi

export TORCH_USE_RTLD_GLOBAL=1
# export OMP_NUM_THREADS=$TOTAL_CORES
# export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES"
echo -e "### using $KMP_SETTING"

function collect_infos {
  LOGFILE=$1
  acc_and_type=(`grep -m 1 "Finally Eval " $LOGFILE | awk '{print $NF}'`)
  acc_loss=(`grep -m 1 "acc_loss is" $LOGFILE | awk '{print $NF}'`)
  fp32_acc=(`grep "fp32 acc is" $LOGFILE | awk '{print $NF}'`)
  acc_type=(`echo $acc_and_type | cut -d ':' -f 1`)
  int8_acc=(`echo $acc_and_type | cut -d ':' -f 2`)
  fp32_perf=(`grep "fp32 performance " $LOGFILE | awk '{print $NF}'`)
  int8_perf=(`grep "int8 performance " $LOGFILE | awk '{print $NF}'`)
  fallback_int8_perf=(`grep "fallback int8 perf " $LOGFILE | awk '{print $NF}'`)
  perf_gain_from_fp32=(`grep "perf gain from fp32 " $LOGFILE | awk '{print $NF}'`)
  perf_loss_from_fallback=(`grep "perf loss from fallback " $LOGFILE | awk '{print $NF}'`)
  fp32_memory_usage=(`grep "fp32 memory usage " $LOGFILE | awk '{print $NF}'`)
  int8_memory_usage=(`grep "int8 memory usage " $LOGFILE | awk '{print $NF}'`)
  
  tunable_op_len=(`grep "tunable op len " $LOGFILE | awk '{print $NF}'`)

  fallback=(`grep  "Fallback layers " $LOGFILE | awk -F ":" '{print $NF}'`)
  
  echo "write_infos={'task_name': '$TASK_NAME', 'model_size': '$MODEL_SIZE', 'data_type': '$DATA_TYPE', 'acc_type': '$acc_type', 'fp32_acc': '$fp32_acc', 'int8_acc': '$int8_acc', 'acc_loss': '$acc_loss','fp32_perf':'$fp32_perf', 'int8_perf':'$int8_perf', 'fallback_int8_perf':'$fallback_int8_perf', 'perf_gain_from_fp32':'$perf_gain_from_fp32', 'perf_loss_from_fallback':'$perf_loss_from_fallback', 'fp32_memory_usage':'$fp32_memory_usage', 'int8_memory_usage':'$int8_memory_usage',  'tunable_op_len':'$tunable_op_len',  'fallback':'$fallback'}" >> final_log
  echo "infos get..."

}

function run_model {
  PREFIX=""
  OUTPUT=$1
  TASK_NAME=$2
  GLUE_DIR=$3
  BATCH_SIZE=$4
  MODEL_SIZE=$5
  DATA_TYPE=$6

  ARGS="$7 $8"
  MAX_SEQ_LENGTH=128
  SCRIPTS=examples/run_glue_tune.py
  if [[ "$TASK_NAME" == "SQuAD" ]]; then
    MAX_SEQ_LENGTH=384
    SCRIPTS=examples/run_squad_tune.py
    # ARGS="$7 $8 --doc_stride 128"    
    ARGS="$7 $8"
  fi

  echo "running model begin..."
  echo "prefix ${ARGS}"
  echo "see $8 is what"
  $PREFIX python $SCRIPTS --model_type bert \
      --model_name_or_path ${OUTPUT} \
      --task_name ${TASK_NAME} \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK_NAME/ \
      --max_seq_length $MAX_SEQ_LENGTH \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --no_cuda \
      --output_dir $OUTPUT $ARGS 2>&1 | tee ${TASK_NAME}_${MODEL_SIZE}_${DATA_TYPE}_log
      # --fallback \
  echo "${TASK_NAME} ${MODEL_SIZE} ${DATA_TYPE}" 
  collect_infos ${TASK_NAME}_${MODEL_SIZE}_${DATA_TYPE}_log

}


for data_type in "int8"
do
  if [[ "$data_type" == "fp32" ]]; then
    ARGS="--mkldnn_eval --do_fp32_inference"
  else
    # ARGS="--do_calibration --fallback"
    ARGS="--tune"
  fi 
  DATA_TYPE=$data_type
  
  for model_size in "large" "base" 
  do
    if [[ "$model_size" == "base" ]]; then
      for task in "MRPC" "CoLA" "STS-B" "SST-2" "RTE" 
      do
          OUTPUT=${GLUE_DIR}/base_weights/${task}_output
          BATCH_SIZE=8
          run_model $OUTPUT $task $GLUE_DIR $BATCH_SIZE $model_size $data_type $ARGS 
      done
    else
      for task in "MRPC" "SQuAD" "QNLI" "RTE" "CoLA" 
      # for task in "SQuAD" "MRPC" "QNLI" "RTE" "CoLA" 
      do
          OUTPUT=${GLUE_DIR}/weights/${task}_output
          BATCH_SIZE=16
          run_model $OUTPUT $task $GLUE_DIR $BATCH_SIZE $model_size $data_type $ARGS 
      done
    fi
  done
done

echo -e "\n### samples/sec = batch_size * it/s\n### batch_size = $BATCH_SIZE"

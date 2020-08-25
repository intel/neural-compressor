###
### script for BERT inference on gpu
### reference:
###   https://github.com/mingfeima/pytorch-transformers#run_gluepy-fine-tuning-on-glue-tasks-for-sequence-classification
###
### 1. prepare dataset:
###   https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
###
### 2. install:
###   pip install --editable .
###   pip install -r ./examples/requirements.txt
###
### 3. use mkldnn for FP32:
###   ./run_inference_cpu.sh --mkldnn_eval (throughput)
###   ./run_inference_cpu.sh --single --mkldnn_eval (realtime)


NUM_CORES=`lscpu | grep Core | awk '{print $4}'`
NUM_THREAD=$NUM_CORES
NUM_NUMA=$((`lscpu | grep 'NUMA node(s)'|awk '{print $3}' ` - 1))
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# ARGS="--do_calibration"
# ARGS="--do_fp32_inference"
# ARGS="--do_int8_inference"
ARGS="--tune"
BATCH_SIZE=16
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  NUM_THREAD=4
  shift
fi
if [[ "$1" == "--mkldnn_eval" ]]; then
  echo "### using mkldnn"
  ARGS="--mkldnn_eval"
fi

echo -e "### using OMP_NUM_THREADS=$NUM_THREAD"
echo -e "### using $KMP_AFFINITY"
echo -e "### using ARGS=$ARGS\n"
for task in "MRPC" 
do
GLUE_DIR=~/glue_data
TASK_NAME=${task}
LOG_DIR=$TASK_NAME"_LOG"
if [ ! -d $LOG_DIR ];then
   mkdir $LOG_DIR
fi
OUTPUT=${GLUE_DIR}/weights/${TASK_NAME}_output
if [[ -d "$OUTPUT" ]]; then
  echo "### using model file from $OUTPUT"
else
  echo -e "\n### model file not found, run fune tune first!\n###  ./run_training_gpu.sh\n"
  exit
fi
INT_PER_NODE=$(($NUM_CORES / $NUM_THREAD - 1 ))
for i in $(seq 0 0)
do
for j in $(seq 0 $INT_PER_NODE)
do
    startid=$(($i*$NUM_CORES+$j*$NUM_THREAD))
    endid=$(($i*$NUM_CORES+$j*$NUM_THREAD+$NUM_THREAD-1))
    OMP_NUM_THREADS=$NUM_THREAD numactl --physcpubind=$startid-$endid --membind=$i \
python ./examples/run_glue_tune.py --model_type bert \
        --model_name_or_path bert-large-uncased \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --no_cuda \
        --output_dir $OUTPUT $ARGS
    
done
done

done

echo -e "\n### samples/sec = batch_size * it/s\n### batch_size = $BATCH_SIZE"

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

TASK_NAME="SQuAD"
# ARGS="--do_int8_inference"
ARGS="--do_ilit_tune"
BATCH_SIZE=16
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  NUM_THREAD=4
  shift
fi
DATA_DIR=~/glue_data/SQuAD
OUTPUT=${DATA_DIR}/../weights/SQuAD_output
if [[ "$1" == "--mkldnn_eval" ]]; then
  echo "### using mkldnn"
  ARGS="--mkldnn_eval"
fi

echo -e "### using OMP_NUM_THREADS=$NUM_THREAD"
echo -e "### using $KMP_AFFINITY"
echo -e "### using ARGS=$ARGS\n"
LOG_DIR=$TASK_NAME"_LOG"
if [ ! -d $LOG_DIR ];then
   mkdir $LOG_DIR
fi
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
python ./examples/question-answering/run_squad_tune.py \
     --model_type bert     \
     --model_name_or_path bert-large-uncased-whole-word-masking   \
     --do_eval \
     --do_lower_case \
     --data_dir $DATA_DIR \
     --max_seq_length 384 \
     --doc_stride 128 \
     --output_dir $OUTPUT \
     --task_name $TASK_NAME \
     --per_gpu_eval_batch_size=$BATCH_SIZE $ARGS
done
done


echo -e "\n### samples/sec = batch_size * it/s\n### batch_size = $BATCH_SIZE"

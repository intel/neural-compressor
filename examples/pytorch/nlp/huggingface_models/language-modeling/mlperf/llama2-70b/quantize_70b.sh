CHECKPOINT_PATH=build/models/Llama2/Llama-2-70b-chat-hf/
CALIBRATION_DATA_PATH=build/preprocessed_data/llama2-70b/open_orca_gpt4_tokenized_llama.calibration_1000.pkl
NUM_GROUPS=-1
NUM_SAMPLES=1000
ITERS=200
BATCH_SIZE=1
NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
END_CORE=$(($NUM_CORES - 1))

run_cmd="python -u"
if (( XPU_COUNT < 1 )); then
        export OMP_NUM_THREADS=$NUM_CORES
        run_cmd="numactl -C 0-$END_CORE python -u"
fi

$run_cmd quantize_autoround.py \
        --model_name ${CHECKPOINT_PATH} \
        --dataset ${CALIBRATION_DATA_PATH} \
        --group_size ${NUM_GROUPS} \
        --bits 4 \
        --iters ${ITERS} \
        --batch_size ${BATCH_SIZE} \
        --device auto \
        --lr 2.5e-3 2>&1 | tee autoround_log_${NUM_GROUPS}g_${NUM_SAMPLES}nsamples_${ITERS}iters.log

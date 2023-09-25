export CUDA_VISIBLE_DEVICES=1
CALIBRATION_DATA=/data4/cyy/gptq_inc/mlperf/data/calibration-data/cnn_dailymail_calibration.json
VALIDATION_DATA=/data4/cyy/gptq_inc/mlperf/data/validation-data/cnn_dailymail_validation.json
MODEL_DIR=/data4/cyy/gptq_inc/mlperf/gpt-j-mlperf/finetuned-gptj/

python -u run_clm.py \
    --model_name_or_path ${MODEL_DIR} \
    --wbits 4 \
    --sym \
    --group_size -1 \
    --nsamples 128 \
    --calib-data-path ${CALIBRATION_DATA} \
    --val-data-path ${VALIDATION_DATA} \
    --calib-iters 128 \
    --prune \
    --target_sparsity 0.3 \
    --use_gpu
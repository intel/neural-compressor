CALIBRATION_DATA=/path/to/your/data/calibration-data/cnn_dailymail_calibration.json
VALIDATION_DATA=/path/to/your/data/validation-data/cnn_dailymail_validation.json
MODEL_DIR=/path/to/finetuned-gptj/

python -u run_gptj_mlperf_int4.py \
    --model_name_or_path ${MODEL_DIR} \
    --wbits 4 \
    --act-order \
    --sym \
    --group_size 128 \
    --nsamples 128 \
    --calib-data-path ${CALIBRATION_DATA} \
    --val-data-path ${VALIDATION_DATA} \
    --calib-iters 128 \

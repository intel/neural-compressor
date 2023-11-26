CALIBRATION_DATA=/your/data/calibration-data/cnn_dailymail_calibration.json
VALIDATION_DATA=/your/data/validation-data/cnn_dailymail_validation.json
MODEL_DIR=/your/gptj/

python -u examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only/run_gptj_mlperf_int4.py \
    --model_name_or_path ${MODEL_DIR} \
    --wbits 4 \
    --sym \
    --group_size -1 \
    --nsamples 128 \
    --calib-data-path ${CALIBRATION_DATA} \
    --val-data-path ${VALIDATION_DATA} \
    --calib-iters 128 \
    --use_max_length \
    --pad_max_length 2048 \
    --use_gpu
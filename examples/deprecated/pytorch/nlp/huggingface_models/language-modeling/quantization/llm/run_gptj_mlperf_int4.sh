CALIBRATION_DATA=/your/data/calibration-data/cnn_dailymail_calibration.json
VALIDATION_DATA=/your/data/validation-data/cnn_dailymail_validation.json
MODEL_DIR=/your/gptj/

python -u examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm/run_gptj_mlperf_int4.py \
    --model_name_or_path ${MODEL_DIR} \
    --wbits 3 \
    --sym \
    --group_size 128 \
    --nsamples 256 \
    --calib-data-path ${CALIBRATION_DATA} \
    --val-data-path ${VALIDATION_DATA} \
    --calib-iters 256 \
    --use_max_length \
    --pad_max_length 2048 \
    --use_gpu

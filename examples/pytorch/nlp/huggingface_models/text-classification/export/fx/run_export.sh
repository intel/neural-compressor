#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
    dtype='fp32'
    quant_format='QDQ' # or QOperator
    for var in "$@"
    do
        case $var in
        --input_model=*)
            input_model=$(echo $var |cut -f2 -d=)
        ;;
        --output_model=*)
            output_model=$(echo $var |cut -f2 -d=)
        ;;
        --dataset_location=*)
            dataset_location=$(echo $var |cut -f2 -d=)
        ;;
        --dtype=*)
            dtype=$(echo $var |cut -f2 -d=)
        ;;
        --quant_format=*)
            quant_format=$(echo $var |cut -f2 -d=)
        ;;
        --approach=*)
            approach=$(echo $var |cut -f2 -d=)
        ;;
        esac
    done

}

# run_tuning
function run_tuning {
    # tuned_checkpoint is used to save torch int8 model.
    tuned_checkpoint=saved_results
    extra_cmd=''
    batch_size=16
    MAX_SEQ_LENGTH=128
    model_name_or_path=${input_model}
    TASK_NAME=${dataset_location}

    python -u ./run_glue.py \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${TASK_NAME} \
        --do_eval \
        --do_train \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ${tuned_checkpoint} \
        --output_model ${output_model} \
        --export \
        --export_dtype ${dtype} \
        --quant_format ${quant_format} \
        --output_dir ${tuned_checkpoint} \
        --overwrite_output_dir \
        --approach ${approach} \
        ${extra_cmd}
}

main "$@"

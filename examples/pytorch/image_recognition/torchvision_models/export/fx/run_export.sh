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
    tuned_checkpoint=saved_results
    for var in "$@"
    do
        case $var in
            --input_model=*)
                input_model=$(echo $var |cut -f2 -d=)
            ;;
            --dataset_location=*)
                dataset_location=$(echo $var |cut -f2 -d=)
            ;;
            --output_model=*)
                output_model=$(echo $var |cut -f2 -d=)
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
    python main.py \
            --pretrained \
            -t \
            -a ${input_model} \
            -b 30 \
            --tuned_checkpoint ${tuned_checkpoint} \
            --output_model ${output_model} \
            --export \
            --export_dtype ${dtype} \
            --quant_format ${quant_format} \
            --approach ${approach} \
            ${dataset_location}

}

main "$@"

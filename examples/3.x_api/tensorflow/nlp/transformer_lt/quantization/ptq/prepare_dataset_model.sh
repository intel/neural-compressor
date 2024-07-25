#!/bin/bash
# set -x

DATA_DIR="../data"
MODEL_DIR="../model"

help()
{
    cat <<- EOF
    Desc: Prepare bert dataset
    -h --help                   help info
    --data_dir                  Output data directory
                                default: './data'
    --model_dir                 Output model directory
                                default: './model'
EOF
    exit 0
}

function main {
    init_params "$@"
    prepare
}

# init params
function init_params {
    for var in "$@"
    do
        case $var in
            --data_dir=*)
                DATA_DIR=$(echo $var |cut -f2 -d=)
            ;;
            --model_dir=*)
                MODEL_DIR=$(echo $var |cut -f2 -d=)
            ;;
            -h|--help) help
            ;;
            *)
            echo "Error: No such parameter: ${var}"
            exit 1
            ;;
        esac
    done
}

# prepare data and model
function prepare {
    if [ ! -d ${DATA_DIR} ]; then
        echo '${DATA_DIR} already exists, please check...'
    fi
    if [ ! -d ${MODEL_DIR} ]; then
        echo '${MODEL_DIR} already exists, please check...'
    fi
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_2_0/transformer-lt-official-fp32-inference.tar.gz
    tar -zxvf transformer-lt-official-fp32-inference.tar.gz
    cd transformer-lt-official-fp32-inference
    tar -zxvf transformer_lt_official_fp32_pretrained_model.tar.gz
    mv transformer_lt_official_fp32_pretrained_model/data ${DATA_DIR}
    mv transformer_lt_official_fp32_pretrained_model/graph ${MODEL_DIR}
}

main "$@"

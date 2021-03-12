#!/bin/bash
# set -x

OUTPUT_DIR="./model"

help()
{
    cat <<- EOF
    Desc: Prepare bert model
    -h --help                   help info
    --output_dir                Output model directory
                                default: './model'
EOF
    exit 0
}

function main {
    init_params "$@"
    convert_model
}

# init params
function init_params {
    for var in "$@"
    do
        case $var in
            --output_dir=*)
                OUTPUT_DIR=$(echo $var |cut -f2 -d=)
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

# convert model
function convert_model {
    if [ ! -d ${OUTPUT_DIR} ]; then
        echo '${OUTPUT_DIR} already exists, please check...'
    fi
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
    unzip bert_large_checkpoints.zip
    mv bert_large_checkpoints ${OUTPUT_DIR}
    
}

main "$@"


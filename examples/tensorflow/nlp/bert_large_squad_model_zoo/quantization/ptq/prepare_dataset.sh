#!/bin/bash
# set -x

OUTPUT_DIR="./data"

help()
{
    cat <<- EOF
    Desc: Prepare bert dataset
    -h --help                   help info
    --output_dir                Output data directory
                                default: './data'
EOF
    exit 0
}

function main {
    init_params "$@"
    convert_dataset
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

# convert dataset
function convert_dataset {
    if [ ! -d ${OUTPUT_DIR} ]; then
        echo '${OUTPUT_DIR} already exists, please check...'
    fi
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
    unzip wwm_uncased_L-24_H-1024_A-16.zip
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
    mv wwm_uncased_L-24_H-1024_A-16 ${OUTPUT_DIR}
    
}

main "$@"


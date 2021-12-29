#!/bin/bash
# set -x

OUTPUT_DIR="./data"
SUBSET="validation"
SHARDS=1

help()
{
    cat <<- EOF
    Desc: Convert prepared raw imagnet dataset to tfrecord
    -h --help                   help info
    --output_dir                Output data directory
                                default: './data'
    --raw_dir                   Raw data directory
    --shards                    Number of shards in TFRecord files.
                                default: '1'
    --subset                    Subset of imagenet, can be validation/train.
                                default: 'validation'
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
            --raw_dir=*)
                RAW_DIR=$(echo $var |cut -f2 -d=)
            ;;
            --shards=*)
                SHARDS=$(echo $var |cut -f2 -d=)
            ;;
            --subset=*)
                SUBSET=$(echo $var |cut -f2 -d=)
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
        mkdir ${OUTPUT_DIR}
    fi
    python imagenet_prepare/build_imagenet_data.py \
        --imagenet_metadata_file "imagenet_prepare/imagenet_metadata.txt" \
        --labels_file "imagenet_prepare/imagenet_lsvrc_2015_synsets.txt" \
        --output_directory ${OUTPUT_DIR} \
        --subset ${SUBSET} \
        --raw_directory ${RAW_DIR} \
        --shards ${SHARDS} 
}

main "$@"


#!/bin/bash
# set -x

OUTPUT_DIR="./model"
# please first prepare_dataset.py
GLUE_DIR="./data"

help()
{
    cat <<- EOF
    Desc: Prepare bert dataset
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
            --dataset_location=*)
                GLUE_DIR=$(echo $var |cut -f2 -d=)
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
function convert_model {
    if [ -d ${OUTPUT_DIR} ]; then
        echo '${OUTPUT_DIR} already exists, please check...'
        exit 0
    fi
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

    python run_classifier.py \
      --task_name=MRPC \
      --do_train=true \
      --do_eval=true \
      --data_dir=$GLUE_DIR/MRPC \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$OUTPUT_DIR
    
    cp $BERT_BASE_DIR/vocab.txt $OUTPUT_DIR/
    cp $BERT_BASE_DIR/bert_config.json $OUTPUT_DIR/
}

main "$@"


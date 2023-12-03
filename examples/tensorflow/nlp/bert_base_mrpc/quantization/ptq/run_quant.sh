#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    python run_classifier.py \
      --task_name=MRPC \
      --data_dir=${dataset_location}/MRPC \
      --vocab_file=${input_model}/vocab.txt \
      --bert_config_file=${input_model}/bert_config.json \
      --init_checkpoint=${input_model}/model.ckpt-343 \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$input_model \
      --output_model=$output_model \
      --tune \

}

main "$@"

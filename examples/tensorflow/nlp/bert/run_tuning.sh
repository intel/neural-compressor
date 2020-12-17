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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    python tune_squad.py \
      --do_predict=True \
      --vocab_file=${dataset_location}/vocab.txt \
      --bert_config_file=${dataset_location}/bert_config.json \
      --predict_file=${dataset_location}/dev-v1.1.json \
      --max_seq_length=384 \
      --doc_stride=128 \
      --output_model=${output_model} \
      --output_dir=./ \
      --mode=tune \
      --predict_batch_size=32 \
      --input_graph=${input_model} \
      # --init_checkpoint=${input_model}/model.ckpt-3649 \

}

main "$@"

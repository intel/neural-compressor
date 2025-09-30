#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --init_checkpoint=*)
          init_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {
    python run_classifier.py \
      --task_name=MRPC \
      --data_dir=${dataset_location}/MRPC \
      --vocab_file=${init_checkpoint}/vocab.txt \
      --bert_config_file=${init_checkpoint}/bert_config.json \
      --init_checkpoint=${init_checkpoint}/model.ckpt-343 \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --eval_batch_size=$batch_size \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$init_checkpoint \
      --input_model=$input_model \
      --mode=$mode \
      --benchmark \

}

main "$@"

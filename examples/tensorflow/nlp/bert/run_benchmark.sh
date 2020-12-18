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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {

    python tune_squad.py \
      --do_predict=True \
      --vocab_file=${dataset_location}/vocab.txt \
      --bert_config_file=${dataset_location}/bert_config.json \
      --predict_file=${dataset_location}/dev-v1.1.json \
      --label_file=${dataset_location}/dev-v1.1.json \
      --max_seq_length=384 \
      --doc_stride=128 \
      --output_dir=./ \
      --mode=${mode} \
      --iters=${iters} \
      --predict_batch_size=${batch_size} \
      --input_graph=${input_model} \
      # --init_checkpoint=${input_model}/model.ckpt-3649 \

}

main "$@"

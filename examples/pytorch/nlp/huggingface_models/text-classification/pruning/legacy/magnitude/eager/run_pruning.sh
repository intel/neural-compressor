#!/bin/bash
set -x

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
      --data_dir=*)
          data_dir=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    if [ "${topology}" = "distilbert_SST-2" ]; then
        python distiller_bert.py \
               --model_type=distilbert \
               --model_name_or_path=distilbert \
               --task_name=SST-2 \
               --do_train \
               --do_eval \
               --do_lower_case \
               --data_dir=${data_dir}/SST-2/ \
               --max_seq_length=128 \
               --per_gpu_train_batch_size=32 \
               --per_gpu_eval_batch_size=16 \
               --learning_rate=5e-5 \
               --num_train_epochs=15.0 \
               --max_grad_norm=1.0 \
               --logging_steps=2105 \
               --save_steps=2105 \
               --output_model=${output_model} \
               --seed=42 \
               --prune \
               --config=${config}
    fi
}

main "$@"

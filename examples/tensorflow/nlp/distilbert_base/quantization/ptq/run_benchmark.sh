#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_benchmark

}

# init params
function init_params {
  # set default value
  input_model="./distilbert_base_fp32.pb"
  dataset_location="./sst2_validation_dataset"
  mode="performance"
  batch_size=128
  max_seq_length=128
  iters=872
  warmup_steps=10
  num_inter=2
  num_intra=28
  benchmark=True
  profile=False

  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --max_seq_length=*)
          max_seq_length=$(echo ${var} |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --warmup_steps=*)
          warmup_steps=$(echo ${var} |cut -f2 -d=)
      ;;
      --num_inter=*)
         num_inter=$(echo ${var} |cut -f2 -d=)
      ;;
      --num_intra=*)
         num_intra=$(echo ${var} |cut -f2 -d=)
      ;;
      --benchmark=*)
         benchmark=$(echo ${var} |cut -f2 -d=)
      ;;
      --profile=*)
         profile=$(echo ${var} |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    cmd="
        python run_inference.py \
            --in-graph=${input_model} \
            --data-location=${dataset_location} \
            --benchmark=${benchmark} \
            --profile=${profile} \
            --mode=${mode} \
            --steps=${iters} \
            --warmup-steps=${warmup_steps} \
            --batch-size=${batch_size} \
            --max-seq-length=${max_seq_length} \
            --num-inter-threads=${num_inter} \
            --num-intra-threads=${num_intra}
        "
    echo $cmd
    eval $cmd
}

main "$@"

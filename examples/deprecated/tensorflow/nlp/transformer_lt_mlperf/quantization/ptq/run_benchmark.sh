#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_benchmark

}

# init params
function init_params {
  # set default value
  input_model="./transformer_mlperf_fp32.pb"
  dataset_location="./transformer_uniform_data"
  file_out="./output_translation_result.txt"
  batch_size=64
  iters=-1
  warmup_steps=10
  bleu_variant="uncased"
  num_inter=2
  num_intra=28

  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --file_out=*)
          file_out=$(echo $var |cut -f2 -d=)
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
      --warmup_steps=*)
          warmup_steps=$(echo ${var} |cut -f2 -d=)
      ;;
      --bleu_variant=*)
          bleu_variant=$(echo ${var} |cut -f2 -d=)
      ;;
      --num_inter=*)
         num_inter=$(echo ${var} |cut -f2 -d=)
      ;;
      --num_intra=*)
         num_intra=$(echo ${var} |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {
    cmd="
        python run_inference.py \
            --input_graph=${input_model} \
            --input_file=${dataset_location}/newstest2014.en \
            --reference_file=${dataset_location}/newstest2014.de \
            --vocab_file=${dataset_location}/vocab.ende.32768 \
            --file_out=${file_out} \
            --benchmark \
            --mode=${mode} \
            --iters=${iters} \
            --warmup_steps=${warmup_steps} \
            --batch_size=${batch_size} \
            --bleu_variant=${bleu_variant} \
            --num_inter=${num_inter} \
            --num_intra=${num_intra}
        "
    echo $cmd
    eval $cmd
}

main "$@"

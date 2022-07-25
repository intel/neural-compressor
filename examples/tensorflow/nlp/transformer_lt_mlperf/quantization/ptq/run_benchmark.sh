#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_benchmark

}

# init params
function init_params {
  # set default value
  topology="transformer_lt_mlperf"
  input_model="./transformer_mlperf_fp32.pb"
  dataset_location="./transformer_uniform_data"
  file_out="./output_translation_result.txt"
  mode="accuracy"
  batch_size=64
  iters=-1
  warmup_steps=5
  bleu_variant="uncased"
  num_inter=2
  num_intra=56

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
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
      *)
          echo "Parameter error: ${var}"
          exit 1
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

#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  iters=100
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
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
      --optimized=*)
          optimized=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --benchmark --iters "${iters}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [[ ${optimized} == "true" ]]; then
        extra_cmd=$extra_cmd" --optimized"
    fi
    echo $extra_cmd
    if [[ "${topology}" == "bert_large_ipex" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        python run_qa.py \
            --model_name_or_path $model_name_or_path \
            --dataset_name squad \
            --do_eval \
            --max_seq_length 384 \
            --no_cuda \
            --output_dir $tuned_checkpoint \
            --per_gpu_eval_batch_size $batch_size \
            $mode_cmd \
            ${extra_cmd}
    fi
    if [[ "${topology}" == "distilbert_base_ipex" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        python run_qa.py \
            --model_name_or_path $model_name_or_path \
            --dataset_name squad \
            --do_eval \
            --max_seq_length 384 \
            --no_cuda \
            --output_dir $tuned_checkpoint \
            --per_gpu_eval_batch_size $batch_size \
            $mode_cmd \
            ${extra_cmd}
    fi
}


main "$@"

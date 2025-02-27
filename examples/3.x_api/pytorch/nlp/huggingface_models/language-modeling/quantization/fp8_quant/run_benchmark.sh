#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  batch_size=1
  tuned_checkpoint=saved_results
  task=lambada_openai

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
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
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
    python_cmd="python"

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --performance"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "opt_125m_fp8" ]; then
        model_name_or_path="facebook/opt-125m"
        tuned_checkpoint="opt_125m_fp8"
    elif [ "${topology}" = "opt_125m_fp8_pcs" ]; then
        model_name_or_path="facebook/opt-125m"
        tuned_checkpoint="opt_125m_fp8_pcs"
    elif [ "${topology}" = "opt_125m_fp8_block_wise" ]; then
        model_name_or_path="facebook/opt-125m"
        tuned_checkpoint="opt_125m_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_8b_fp8" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8"
    elif [ "${topology}" = "llama3_1_8b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_8b_fp8_block_wise_pcs" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8_block_wise_pcs"
    elif [ "${topology}" = "llama2_70b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama2/Llama-2-70b-hf/"
        tuned_checkpoint="/software/llama_fp8/llama2_70b_fp8_block_wise"
    elif [ "${topology}" = "mixtral_8x7b_fp8_block_wise" ]; then
        model_name_or_path="mistralai/Mixtral-8x7B-v0.1"
        tuned_checkpoint="/software/mixtral_fp8/mixtral_8x7b_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_405b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-405B-Instruct/"
        tuned_checkpoint="/software/llama_fp8/llama3_1_405b_fp8_block_wise"
        python_cmd="deepspeed --num_gpus 8"
    fi

    if [[ ${int8} == "true" ]]; then
        ${python_cmd} quantize.py \
            --model ${tuned_checkpoint} \
            --load\
            --task ${task} \
            --batch_size ${batch_size} \
            --use_hpu_graph \
            ${mode_cmd}
    else
        ${python_cmd} quantize.py \
            --model ${model_name_or_path} \
            --task ${task} \
            --batch_size ${batch_size} \
            --use_hpu_graph \
            ${mode_cmd}
    fi
}

main "$@"

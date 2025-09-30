#!/bin/bash
set -x

function main {
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --tasks=*)
          tasks=$(echo $var |cut -f2 -d=)
      ;;
      --tp_size=*)
          tp_size=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    extra_model_args=""
    extra_cmd=""
    batch_size=${batch_size:=1}

    if [ "${topology}" = "llama4_mxfp4" ]; then
        extra_model_args="max_model_len=8192,max_num_seqs=1024,max_gen_toks=2048,kv_cache_dtype=auto,gpu_memory_utilization=0.7"
        extra_cmd="--gen_kwargs max_gen_toks=2048"
    fi

    if [[ "${tasks}" == *"chartqa"* || "${tasks}" == *"mmmu_val"* ]]; then
        model="vllm-vlm"
        extra_cmd=${extra_cmd}" --apply_chat_template"
    else
        model="vllm"
    fi

    NCCL_NVLS_ENABLE=0 VLLM_USE_STANDALONE_COMPILE=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    lm_eval --model ${model} \
            --model_args pretrained=${input_model},tensor_parallel_size=${tp_size},${extra_model_args},enable_expert_parallel=True \
            --tasks ${tasks} \
            --batch_size ${batch_size} \
            ${extra_cmd}
}

main "$@"

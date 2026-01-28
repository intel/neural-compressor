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
      --static_kv_dtype=*)
          kv_cache_dtype=$(echo $var |cut -f2 -d=)
      ;;
      --static_attention_dtype=*)
          attention_dtype=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    extra_model_args=""
    extra_cmd=""
    kv_cache_dtype=${kv_cache_dtype:="auto"}
    attention_dtype=${attention_dtype:="auto"}
    batch_size=${batch_size:=1}

    if [ "${topology}" = "llama4_mxfp4" ]; then
        export VLLM_AR_MXFP4_MODULAR_MOE=1
        export VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
        export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
        export VLLM_ENABLE_STATIC_MOE=0
        export VLLM_USE_DEEP_GEMM=0
        export VLLM_ENABLE_AR_EXT=1
        extra_model_args="max_model_len=8192,max_num_seqs=1024,max_gen_toks=2048,gpu_memory_utilization=0.7"
        extra_cmd="--gen_kwargs max_gen_toks=2048"
    fi

    if [[ "${tasks}" == *"chartqa"* || "${tasks}" == *"mmmu_val"* ]]; then
        model="vllm-vlm"
        extra_cmd=${extra_cmd}" --apply_chat_template"
    elif [[ "${tasks}" == *"longbench"* ]]; then
	    model="vllm"
        extra_cmd="--seed 42 --apply_chat_template --gen_kwargs {\"temperature\":0.0} "
	    extra_model_args="max_model_len=66000,gpu_memory_utilization=0.7"
    else
        model="vllm"
    fi

    if [[ "${kv_cache_dtype}" == "fp8" ]]; then
        export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
        export VLLM_ATTENTION_BACKEND="FLASHINFER"
        echo "Using FP8 for KV cache"
    fi

    if [[ "${attention_dtype}" == "fp8" ]]; then
        export VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0
        export VLLM_ATTENTION_BACKEND="FLASHINFER"
        kv_cache_dtype="fp8"
        echo "Using FP8 Attention"
    fi

    NCCL_NVLS_ENABLE=0 VLLM_USE_STANDALONE_COMPILE=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    lm_eval --model ${model} \
            --model_args pretrained=${input_model},tensor_parallel_size=${tp_size},${extra_model_args},enable_expert_parallel=True,kv_cache_dtype=${kv_cache_dtype} \
            --tasks ${tasks} \
            --batch_size ${batch_size} \
            ${extra_cmd}
}

main "$@"

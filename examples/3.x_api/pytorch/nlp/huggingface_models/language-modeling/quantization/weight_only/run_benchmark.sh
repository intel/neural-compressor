#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16
  tuned_checkpoint=saved_results
  task=lambada_openai
  echo ${max_eval_samples}
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
    extra_cmd=''

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --performance --iters "${iters}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --load"
    fi
    echo $extra_cmd

    if [ "${topology}" = "opt_125m_woq_gptq_int4" ]; then
        model_name_or_path="facebook/opt-125m"
    elif [ "${topology}" = "opt_125m_woq_gptq_nf4_dq_bnb" ]; then
        model_name_or_path="facebook/opt-125m"
    elif [ "${topology}" = "opt_125m_woq_gptq_int4_dq_ggml" ]; then
        model_name_or_path="facebook/opt-125m"
    elif [ "${topology}" = "llama2_7b_gptq_int4" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
    elif [ "${topology}" = "llama2_7b_gptq_nf4_dq_bnb" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
    elif [ "${topology}" = "llama2_7b_gptq_int4_dq_ggml" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_rtn_nf4_dq_bnb" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4_dq_ggml" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_gptq_nf4_dq_bnb" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4_dq_ggml" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "gpt_j_woq_awq_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
    elif [ "${topology}" = "opt_125m_woq_awq_int4" ]; then
        model_name_or_path="facebook/opt-125m"
    elif [ "${topology}" = "opt_125m_woq_autoround_int4" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --woq_algo AutoRound"
    elif [ "${topology}" = "opt_125m_woq_autoround_int4_hpu" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --woq_algo AutoRound"
    elif [ "${topology}" = "opt_125m_woq_autotune_int4" ]; then
        model_name_or_path="facebook/opt-125m"
    fi

    if [[ ${mode} == "accuracy" ]]; then
        python -u run_clm_no_trainer.py \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --task ${task} \
            --batch_size ${batch_size} \
            ${extra_cmd} ${mode_cmd}
    elif [[ ${mode} == "performance" ]]; then
        incbench --num_cores_per_instance 4 run_clm_no_trainer.py \
            --model ${model_name_or_path} \
            --batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            ${extra_cmd} ${mode_cmd}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi
        
}

main "$@"

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
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
       --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
       ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=''
    batch_size=8
    tuned_checkpoint="saved_results"
    python_cmd="python"

    if [ "${topology}" = "opt_125m_fp8" ]; then
        model_name_or_path="facebook/opt-125m"
        tuned_checkpoint="opt_125m_fp8"
    elif [ "${topology}" = "opt_125m_fp8_pcs" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --scale_method act_maxabs_pow2_weights_pcs_opt_pow2"
        tuned_checkpoint="opt_125m_fp8_pcs"
    elif [ "${topology}" = "opt_125m_fp8_block_wise" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration"
        tuned_checkpoint="opt_125m_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_8b_fp8" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8"
    elif [ "${topology}" = "llama3_1_8b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_8b_fp8_block_wise_pcs" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration --scale_method act_maxabs_pow2_weights_pcs_opt_pow2"
        tuned_checkpoint="/software/llama_fp8/llama3_1_8b_fp8_block_wise_pcs"
    elif [ "${topology}" = "llama2_70b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama2/Llama-2-70b-hf/"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration"
        tuned_checkpoint="/software/llama_fp8/llama2_70b_fp8_block_wise"
    elif [ "${topology}" = "mixtral_8x7b_fp8_block_wise" ]; then
        model_name_or_path="mistralai/Mixtral-8x7B-v0.1"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration"
        tuned_checkpoint="/software/mixtral_fp8/mixtral_8x7b_fp8_block_wise"
    elif [ "${topology}" = "llama3_1_405b_fp8_block_wise" ]; then
        model_name_or_path="/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-405B-Instruct/"
        extra_cmd=$extra_cmd" --enable_block_wise_calibration"
        tuned_checkpoint="/software/llama_fp8/llama3_1_405b_fp8_block_wise"
        python_cmd="deepspeed --num_gpus 8"
    fi

    ${python_cmd} quantize.py \
        --model_name_or_path ${model_name_or_path} \
        --quantize \
        --use_hpu_graph \
        --batch_size ${batch_size} \
        --save \
        --save_path ${tuned_checkpoint} \
        ${extra_cmd}
}

main "$@"

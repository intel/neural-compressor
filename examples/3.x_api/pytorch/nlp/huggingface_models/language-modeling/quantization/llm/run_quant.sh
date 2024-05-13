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
    approach='static'
    DATASET_NAME="NeelNanda/pile-10k"
    tuned_checkpoint="saved_results"

    if [ "${topology}" = "opt_125m_woq_gptq_int4" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
    elif [ "${topology}" = "opt_125m_woq_gptq_int4_dq_bnb" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
        extra_cmd=$extra_cmd" --double_quant_type BNB_NF4"
    elif [ "${topology}" = "opt_125m_woq_gptq_int4_dq_ggml" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length --gptq_percdamp 0.1 --gptq_actorder"
        extra_cmd=$extra_cmd" --double_quant_type GGML_TYPE_Q4_K"
    elif [ "${topology}" = "opt_125m_ipex" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex"
    elif [ "${topology}" = "opt_125m_ipex_sq" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 0.5"
    elif [ "${topology}" = "llama2_7b_gptq_int4" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
    elif [ "${topology}" = "llama2_7b_gptq_int4_dq_bnb" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
        extra_cmd=$extra_cmd" --double_quant_type BNB_NF4"
    elif [ "${topology}" = "llama2_7b_gptq_int4_dq_ggml" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
        extra_cmd=$extra_cmd" --double_quant_type GGML_TYPE_Q4_K"
    elif [ "${topology}" = "llama2_7b_ipex" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        extra_cmd=$extra_cmd" --ipex"
    elif [ "${topology}" = "llama2_7b_ipex_sq" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 0.8"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo RTN --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4_dq_bnb" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo RTN --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search"
        extra_cmd=$extra_cmd" --double_quant_type BNB_NF4"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4_dq_ggml" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo RTN --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search"
        extra_cmd=$extra_cmd" --double_quant_type GGML_TYPE_Q4_K"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4_dq_bnb" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
        extra_cmd=$extra_cmd" --double_quant_type BNB_NF4"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4_dq_ggml" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_use_mse_search --gptq_use_max_length"
        extra_cmd=$extra_cmd" --double_quant_type GGML_TYPE_Q4_K"
    elif [ "${topology}" = "gpt_j_ipex" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        extra_cmd=$extra_cmd" --ipex"
    elif [ "${topology}" = "gpt_j_ipex_sq" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 1.0"
    fi

    python -u run_clm_no_trainer.py \
        --model ${model_name_or_path} \
        --dataset ${DATASET_NAME} \
        --quantize \
        --accuracy \
        --approach ${approach} \
        --output_dir ${tuned_checkpoint} \
        --tasks "lambada_openai" \
        --batch_size ${batch_size} \
        ${extra_cmd}
}

main "$@"

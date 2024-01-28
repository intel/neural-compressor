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

    if [ "${topology}" = "opt_125m_woq_awq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo AWQ --calib_iters 128"
    elif [ "${topology}" = "opt_125m_woq_gptq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ"
    elif [ "${topology}" = "opt_125m_woq_gptq_int4" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_scheme asym --woq_group_size 128 --gptq_use_max_length"
    elif [ "${topology}" = "opt_125m_woq_teq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo TEQ"
    elif [ "${topology}" = "opt_125m_ipex" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --int8_bf16_mixed"
    elif [ "${topology}" = "opt_125m_ipex_sq" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --int8_bf16_mixed --sq --alpha 0.5"
    elif [ "${topology}" = "bloom_560m_ipex_sq" ]; then
        model_name_or_path="bigscience/bloom-560m"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 0.5"
    elif [ "${topology}" = "llama2_7b_ipex_sq" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        extra_cmd=$extra_cmd" --ipex --sq --alpha auto"
    elif [ "${topology}" = "gpt_j_ipex_sq" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 1.0"
    elif [ "${topology}" = "gpt_j_woq_rtn_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo RTN --woq_bits 4 --woq_group_size 128 --woq_scheme asym --woq_enable_mse_search"
    elif [ "${topology}" = "gpt_j_woq_gptq_int4" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --gptq_use_max_length"
    elif [ "${topology}" = "falcon_7b_sq" ]; then
        model_name_or_path="tiiuae/falcon-7b-instruct"
        extra_cmd=$extra_cmd" --sq --alpha 0.5"
    elif [ "${topology}" = "falcon_7b_woq_gptq_int4" ]; then
        model_name_or_path="tiiuae/falcon-7b-instruct"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ --woq_bits 4 --woq_group_size 128 --woq_scheme asym --gptq_use_max_length"
    fi

    python -u run_clm_no_trainer.py \
        --model ${model_name_or_path} \
        --dataset ${DATASET_NAME} \
        --approach ${approach} \
        --output_dir ${tuned_checkpoint} \
        --quantize \
        --batch_size ${batch_size} \
        ${extra_cmd}

}

main "$@"

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
  approach=static
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
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    if [ "${topology}" = "opt_125m_woq_awq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo AWQ --calib_iters 128"
    elif [ "${topology}" = "opt_125m_woq_gptq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo GPTQ"
    elif [ "${topology}" = "opt_125m_woq_teq" ]; then
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --woq_algo TEQ"
    elif [ "${topology}" = "opt_125m" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --int8_bf16_mixed"
    elif [ "${topology}" = "opt_125m_sq" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --int8_bf16_mixed --sq --alpha 0.5"
    elif [ "${topology}" = "llama_7b_sq" ]; then
        model_name_or_path="decapoda-research/llama-7b-hf"
        extra_cmd=$extra_cmd" --ipex --sq --alpha auto"
    elif [ "${topology}" = "gpt_j_sq" ]; then
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 1.0"
    elif [ "${topology}" = "falcon_7b_sq" ]; then
        model_name_or_path="tiiuae/falcon-7b-instruct"
        extra_cmd=$extra_cmd" --sq --alpha 0.5"
    fi

    python -u run_clm_no_trainer.py \
        --model ${model_name_or_path} \
        --approach ${approach} \
        --output_dir ${tuned_checkpoint} \
        --task ${task} \
        ${extra_cmd} ${mode_cmd}
}

main "$@"

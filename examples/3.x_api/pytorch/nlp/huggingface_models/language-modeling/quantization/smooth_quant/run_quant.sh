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

    if [ "${topology}" = "opt_125m_ipex_sq" ]; then
        model_name_or_path="facebook/opt-125m"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 0.5"
    elif [ "${topology}" = "llama2_7b_ipex_sq" ]; then
        model_name_or_path="meta-llama/Llama-2-7b-hf"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 0.8"
    elif [ "${topology}" = "gpt_j_ipex_sq" ]; then
        model_name_or_path="EleutherAI/gpt-j-6b"
        extra_cmd=$extra_cmd" --ipex --sq --alpha 1.0"
    fi

    python -u run_clm_no_trainer.py \
        --model ${model_name_or_path} \
        --dataset ${DATASET_NAME} \
        --quantize \
        --approach ${approach} \
        --output_dir ${tuned_checkpoint} \
        --tasks "lambada_openai" \
        --batch_size ${batch_size} \
        ${extra_cmd}
}

main "$@"

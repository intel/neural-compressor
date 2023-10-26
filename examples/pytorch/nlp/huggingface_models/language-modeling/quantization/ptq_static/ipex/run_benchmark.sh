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
  lm_eval_tasks="lambada_openai  piqa"
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
       --task=*)
           task=$(echo $var |cut -f2 -d=)
       ;;
       --approach=*)
           approach=$(echo $var |cut -f2 -d=)
       ;;
       --backend=*)
           backend=$(echo $var |cut -f2 -d=)
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

    if [ "${topology}" = "gpt_neo" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
    elif [ "${topology}" = "gpt_j" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        if [ "${backend}" = "ipex" ]; then
            script="run_clm_no_trainer.py"
            model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "gpt_j_woq" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        lm_eval_tasks="lambada_openai"
        extra_cmd=$extra_cmd" --approach weight_only"
   elif [ "${topology}" = "chatglm_woq" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="THUDM/chatglm-6b"
        lm_eval_tasks="lambada_openai"
        extra_cmd=$extra_cmd" --approach weight_only"
    elif [ "${topology}" = "gpt_j_woq_awq" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        lm_eval_tasks="lambada_openai"
        extra_cmd=$extra_cmd" --approach weight_only"
    elif [ "${topology}" = "mpt_7b_chat" ]; then
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        model_name_or_path="mosaicml/mpt-7b-chat"
    elif [ "${topology}" = "falcon_7b_instruct" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="tiiuae/falcon-7b-instruct"
    elif [ "${topology}" = "opt_125m_woq"  -o \
           "${topology}" = "opt_125m_woq_awq"  -o \
           "${topology}" = "opt_125m_woq_gptq"  -o \
           "${topology}" = "opt_125m_woq_teq" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="facebook/opt-125m"
        lm_eval_tasks="lambada_openai"
        extra_cmd=$extra_cmd" --approach weight_only"
    elif [ "${topology}" = "opt_125m" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="facebook/opt-125m"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "opt_1.3b" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="facebook/opt-1.3b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "opt_2.7b" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="facebook/opt-2.7b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "opt_6.7b" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="facebook/opt-6.7b"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "llama_7b" ]; then
        script="run_clm_no_trainer.py"
        model_name_or_path="decapoda-research/llama-7b-hf"
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
    elif [ "${topology}" = "bert" ]; then
        script="run_mlm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
    elif [ "${topology}" = "xlnet" ]; then
        script="run_plm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
    elif [ "${topology}" = "gpt_neox" ]; then
        script="run_clm.py"
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
    elif [ "${topology}" = "bloom" ]; then
        script="run_clm.py"
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
    fi
    
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        if [ ${script} != "run_clm_no_trainer.py" ]; then
            model_name_or_path=${tuned_checkpoint}
        fi
    fi


    if [ "${script}" == "run_clm_no_trainer.py" ]; then
        if [ "${lm_eval_tasks}" != "" ]; then
	    extra_cmd=$extra_cmd" --tasks ${lm_eval_tasks}"
        fi
    fi

    echo $extra_cmd
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "benchmark" ]]; then
	if [ "${script}" == "run_clm_no_trainer.py" ]; then
            echo "Error: Only support accuracy now."
            echo "Please go to text-generation folder to get performance."
            exit 1
	 fi
	 mode_cmd=" --benchmark"

    fi


    if [ "${script}" == "run_clm_no_trainer.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --batch_size ${batch_size} \
            ${mode_cmd} \
            ${extra_cmd}
    elif [ -z ${DATASET_CONFIG_NAME} ];then
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ./tmp/benchmark_output \
            --overwrite_output_dir \
            --overwrite_cache \
            --no_cuda \
            ${mode_cmd} \
            ${extra_cmd}
    else
        python -u ${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG_NAME} \
            --do_eval \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ./tmp/benchmark_output \
            --overwrite_output_dir \
            --overwrite_cache \
            --no_cuda \
            ${mode_cmd} \
            ${extra_cmd}
    fi
}

main "$@"

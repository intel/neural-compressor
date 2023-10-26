#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="gpt"
  tuned_checkpoint="saved_results"
  DATASET_NAME="wikitext"
  model_name_or_path="EleutherAI/gpt-neo-125m"
  extra_cmd=""
  batch_size=8
  model_type="bert"
  approach="PostTrainingStatic"
  alpha=0.5
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

# run_tuning
function run_tuning {
    if [ "${topology}" = "" ]; then
        script="run_clm.py"
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125m"
        task="clm"
        approach="PostTrainingStatic"
        backend=""
    elif [ "${topology}" = "gpt_neo" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="EleutherAI/gpt-neo-125M"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "gpt_j" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
        if [ "${backend}" = "ipex" ]; then
                extra_cmd=$extra_cmd" --ipex"
                script="run_clm_no_trainer.py"
                DATASET_NAME="NeelNanda/pile-10k"
                model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
                approach="PostTrainingStatic"
		extra_cmd=$extra_cmd" --int8_bf16_mixed"
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
        fi
    elif [ "${topology}" = "gpt_j_woq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only"
    elif [ "${topology}" = "chatglm_woq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="THUDM/chatglm-6b"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only --woq_algo RTN"
    elif [ "${topology}" = "gpt_j_woq_awq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="/tf_dataset2/models/pytorch/gpt-j-6B"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only  --woq_algo AWQ --calib_iters 128"
    elif [ "${topology}" = "mpt_7b_chat" ]; then
	if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="mosaicml/mpt-7b-chat"
        approach="PostTrainingStatic"
	    alpha=0.95
    elif [ "${topology}" = "falcon_7b_instruct" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="tiiuae/falcon-7b-instruct"
        approach="PostTrainingStatic"
        alpha=0.7
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "opt_125m_woq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only --woq_algo RTN --woq_enable_mse_search"
    elif [ "${topology}" = "opt_125m_woq_awq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only --woq_algo AWQ --calib_iters 128"
    elif [ "${topology}" = "opt_125m_woq_gptq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only --woq_algo GPTQ"
    elif [ "${topology}" = "opt_125m_woq_teq" ]; then
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-125m"
        approach="weight_only"
        extra_cmd=$extra_cmd" --approach weight_only --woq_algo TEQ"
    elif [ "${topology}" = "opt_125m" ]; then
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-125m"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --int8_bf16_mixed"
	    alpha=0.8
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "opt_1.3b" ]; then
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-1.3b"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --int8_bf16_mixed"
	    alpha=0.8
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "opt_2.7b" ]; then
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-2.7b"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --int8_bf16_mixed"
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "opt_6.7b" ]; then
        if [ "${backend}" = "ipex" ]; then
            extra_cmd=$extra_cmd" --ipex"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="facebook/opt-6.7b"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --int8_bf16_mixed"
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "llama_7b" ]; then
        if [ "${backend}" = "ipex" ]; then
            alpha=0.8
            extra_cmd=$extra_cmd" --ipex"
	    extra_cmd=$extra_cmd" --calib_iters 100"
        fi
        script="run_clm_no_trainer.py"
        DATASET_NAME="NeelNanda/pile-10k"
        model_name_or_path="decapoda-research/llama-7b-hf"
        approach="PostTrainingStatic"
        extra_cmd=$extra_cmd" --sq --alpha "${alpha}
    elif [ "${topology}" = "bert" ]; then
        if [ "${task}" = "mlm" ]; then
            script="run_mlm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="bert-base-uncased"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --metric_for_best_model accuracy \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "xlnet" ]; then
        if [ "${task}" = "plm" ]; then
            script="run_plm.py"
        fi
        DATASET_NAME="wikitext"
        DATASET_CONFIG_NAME="wikitext-2-raw-v1"
        model_name_or_path="xlnet-base-cased"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        elif [ "${approach}" = "qat" ]; then
            approach="QuantizationAwareTraining"
            extra_cmd=$extra_cmd" --learning_rate 1e-5 \
                    --num_train_epochs 6 \
                    --eval_steps 100 \
                    --save_steps 100 \
                    --greater_is_better True \
                    --load_best_model_at_end True \
                    --evaluation_strategy steps \
                    --save_strategy steps \
                    --metric_for_best_model accuracy \
                    --save_total_limit 1"
        fi
    elif [ "${topology}" = "gpt_neox" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="oscar"
        DATASET_CONFIG_NAME="unshuffled_original_ast"
        model_name_or_path="abeja/gpt-neox-japanese-2.7b"
        if [ "${approach}" = "dynamic" ]; then
            approach="PostTrainingDynamic"
        elif [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
    elif [ "${topology}" = "bloom" ]; then
        if [ "${task}" = "clm" ]; then
            script="run_clm.py"
        fi
        DATASET_NAME="lambada"
        model_name_or_path="bigscience/bloom-560m"
        if [ "${approach}" = "static" ]; then
            approach="PostTrainingStatic"
        fi
        extra_cmd=$extra_cmd" --smooth_quant --sampling_size 400 --torchscript"
    fi

    if [ "${script}" = "run_clm_no_trainer.py" ];then
        python -u ./${script} \
            --model ${model_name_or_path} \
            --output_dir ${tuned_checkpoint} \
            --dataset ${DATASET_NAME} \
            --quantize \
            ${extra_cmd}
    elif [ -z ${DATASET_CONFIG_NAME} ];then
        python -u ./${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --quantization_approach ${approach} \
            ${extra_cmd}
    else
        python -u ./${script} \
            --model_name_or_path ${model_name_or_path} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG_NAME} \
            --do_eval \
            --do_train \
            --per_device_eval_batch_size ${batch_size} \
            --output_dir ${tuned_checkpoint} \
            --no_cuda \
            --tune \
            --overwrite_output_dir \
            --overwrite_cache \
            --quantization_approach ${approach} \
            ${extra_cmd}
    fi
}

main "$@"

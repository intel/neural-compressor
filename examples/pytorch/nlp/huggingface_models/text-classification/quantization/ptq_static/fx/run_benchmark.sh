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
  tuned_checkpoint=int8_model_dir
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
    MAX_SEQ_LENGTH=128
    TASK_NAME='mrpc'
    model_name_or_path=${input_model}
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --performance --iters "${iters}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_base_MRPC" ];then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_CoLA" ]; then
        TASK_NAME='cola'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_STS-B" ]; then
        TASK_NAME='stsb'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_SST-2" ]; then
        TASK_NAME='sst2'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_base_RTE" ]; then
        TASK_NAME='rte'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_MRPC" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_QNLI" ]; then
        TASK_NAME='qnli'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='rte'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "bert_large_CoLA" ]; then
        TASK_NAME='cola'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "funnel_MRPC_fx" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "distilbert_base_MRPC_fx" ]; then
        TASK_NAME='mrpc'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "xlm-roberta-base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "flaubert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "barthez_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "longformer_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "layoutlm_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
        model_type='layoutlm'
    elif [ "${topology}" = "deberta_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "squeezebert_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "xlnet_base_cased_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "roberta_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    elif [ "${topology}" = "camembert_base_MRPC" ]; then
        TASK_NAME='MRPC'
        model_name_or_path=${input_model}
    fi

    extra_cmd='--model_name_or_path '${input_model}
    if [[ ${int8} == "true" ]]; then
        extra_cmd='--model_name_or_path '${tuned_checkpoint}
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    python -u run_glue.py \
        --task_name ${TASK_NAME} \
        --do_eval \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --no_cuda \
        --output_dir ./output_log \
        --overwrite_output_dir \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"

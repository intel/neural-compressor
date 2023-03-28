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
    extra_cmd='--model_name_or_path '${input_model}

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --performance --iters "${iters}
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "bert_large_RTE" ]; then
        TASK_NAME='rte'
    elif [ "${topology}" = "bert_base_CoLA" ]; then
        TASK_NAME='cola'
    elif [ "${topology}" = "bert_base_STS-B" ]; then
        TASK_NAME='stsb'
    elif [ "${topology}" = "bert_base_SST-2" ]; then
        TASK_NAME='sst2'
    elif [ "${topology}" = "bert_base_RTE" ]; then
        TASK_NAME='rte'
    elif [ "${topology}" = "bert_large_QNLI" ]; then
        TASK_NAME='qnli'
    elif [ "${topology}" = "bert_large_CoLA" ]; then
        TASK_NAME='cola'
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd='--model_name_or_path '${tuned_checkpoint}
        extra_cmd=$extra_cmd" --int8"
    else
        extra_cmd='--model_name_or_path '${input_model}
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

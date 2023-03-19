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

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --performance "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "pegasus_samsum" ]; then
        model_name_or_path='lvwerra/pegasus-samsum'
        extra_cmd='--dataset_name samsum'
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    python -u run_summarization.py \
        --model_name_or_path ${model_name_or_path} \
        --do_eval \
        --predict_with_generate \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ${tuned_checkpoint} \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"

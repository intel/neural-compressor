#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
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

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd="--accuracy "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd="--performance "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [ -n "$dataset_location" ];then
        extra_cmd=$extra_cmd"--dataset_dir ${dataset_location} "
    fi
    if [ -n "$input_model" ];then
        extra_cmd=$extra_cmd"--pytorch_checkpoint ${input_model} "
    fi
    if [ -n "$tuned_checkpoint" ];then
        extra_cmd=$extra_cmd"--tuned_checkpoint ${tuned_checkpoint} "
    fi
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd"--int8"
    fi

    python run_tune.py \
                    --backend pytorch \
                    --manifest $dataset_location/dev-clean-wav.json \
                    --pytorch_config_toml pytorch/configs/rnnt.toml \
                    --scenario SingleStream \
                    ${mode_cmd} \
                    ${extra_cmd}
}

main "$@"

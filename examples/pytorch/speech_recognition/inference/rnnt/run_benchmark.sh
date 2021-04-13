#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  output_model=saved_results
  for var in "$@"
  do
    case $var in
      --dataset=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --log_dir=*)
          log_dir=$(echo $var |cut -f2 -d=)
      ;;
      --output_dir=*)
          output_dir=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done
  mkdir -p $log_dir $output_dir
}

# run_benchmark
function run_benchmark {
    extra_cmd=""
    if [ -n "$dataset" ];then
        extra_cmd=$extra_cmd"--dataset_dir ${dataset} "
    fi
    if [ -n "$input_model" ];then
        extra_cmd=$extra_cmd"--pytorch_checkpoint ${input_model} "
    fi
    if [ -n "$log_dir" ];then
        extra_cmd=$extra_cmd"--log_dir ${log_dir} "
    fi
    if [ -n "$output_dir" ];then
        extra_cmd=$extra_cmd"--tuned_checkpoint ${output_dir} "
    fi

    python run_tune.py \
                    --backend pytorch \
                    --manifest $dataset/dev-clean-wav.json \
                    --pytorch_config_toml pytorch/configs/rnnt.toml \
                    --scenario Offline \
                    --benchmark \
                    --int8 \
                    ${extra_cmd}
}

main "$@"

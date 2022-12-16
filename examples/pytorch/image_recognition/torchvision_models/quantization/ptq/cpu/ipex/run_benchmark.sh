#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=32
  input_model=saved_results
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
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iter ${iters} --performance "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [ "resnext101_32x16d_wsl_ipex" = "${topology}" ];then
        extra_cmd=$extra_cmd" --hub"
    fi
    extra_cmd=$extra_cmd" ${dataset_location}"
    echo $extra_cmd


    python main.py \
            --pretrained \
            --tuned_checkpoint ${input_model} \
            -b ${batch_size} \
            -a $topology \
            ${mode_cmd} \
            ${dataset_location}
}

main "$@"

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
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --iter ${iters} --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [ "resnext101_32x16d_wsl_ipex" = "${topology}" ];then
        extra_cmd=$extra_cmd" --hub"
    fi
    result=$(echo $topology | grep "ipex")
    if [[ "$result" != "" ]];then
        sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location/train|g" conf_ipex.yaml
        sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location/val|g" conf_ipex.yaml
        if [[ ${int8} == "true" ]]; then
            extra_cmd=$extra_cmd" --int8"
        fi
        extra_cmd=$extra_cmd" --ipex"
        topology=${topology%*${topology:(-5)}}
    else
        sed -i "/\/path\/to\/calibration\/dataset/s|root:.*|root: $dataset_location/train|g" conf.yaml
        sed -i "/\/path\/to\/evaluation\/dataset/s|root:.*|root: $dataset_location/val|g" conf.yaml
        if [[ ${int8} == "true" ]]; then
            extra_cmd=$extra_cmd" --int8"
        fi
    fi
    extra_cmd=$extra_cmd" ${dataset_location}"
    echo $extra_cmd

    python main.py \
            --pretrained \
            --tuned_checkpoint ${tuned_checkpoint} \
            -b ${batch_size} \
            -a $topology \
            ${mode_cmd} \
            ${extra_cmd}
}

main "$@"

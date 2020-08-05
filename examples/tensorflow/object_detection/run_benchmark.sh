#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {
    if [ "$topology" = "ssd_resnet50_v1" ];then
        config_file='ssd_resnet50_v1.yaml'
    fi

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy-only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --iter ${iters} --benchmark"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    python  infer_detections.py \
            --batch-size ${batch_size} \
            --input-graph ${input_model} \
            --data-location ${dataset_location} \
            --config ${config_file} \
             ${mode_cmd}
}

main "$@"
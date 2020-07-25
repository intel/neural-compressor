#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --model_location=*)
          model_location=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
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
    extra_cmd=''
    batch_size=64
    num_inference_batches=500
    dataset=${dataset_location}
    ctx='cpu'

    if [ "${topology}" = "resnet50_v1" ];then
        symbol_file=${model_location}'/resnet50_v1-symbol.json'
        param_file=${model_location}'/resnet50_v1-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'
    elif [ "${topology}" = "squeezenet1.0" ]; then
        symbol_file=${model_location}'/squeezenet1.0-symbol.json'
        param_file=${model_location}'/squeezenet1.0-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'
    elif [ "${topology}" = "mobilenet1.0" ]; then
        symbol_file=${model_location}'/mobilenet1.0-symbol.json'
        param_file=${model_location}'/mobilenet1.0-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'

    elif [ "${topology}" = "mobilenetv2_1.0" ]; then
        symbol_file=${model_location}'/mobilenetv2_1.0-symbol.json'
        param_file=${model_location}'/mobilenetv2_1.0-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'

    elif [ "${topology}" = "inceptionv3" ]; then
        symbol_file=${model_location}'/inceptionv3-symbol.json'
        param_file=${model_location}'/inceptionv3-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --image-shape=3,299,299'

    elif [ "${topology}" = "resnet18_v1" ]; then
        symbol_file=${model_location}'/resnet18_v1-symbol.json'
        param_file=${model_location}'/resnet18_v1-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'
    fi

    python -u imagenet_inference.py \
            --symbol-file=${symbol_file} \
            --param-file=${param_file} \
            --batch-size=${batch_size} \
            --num-inference-batches=${num_inference_batches} \
            --dataset=${dataset} \
            --ctx=${ctx} \
            --tune \
            --output-graph=${output_model} \
            ${extra_cmd}

}

main "$@"
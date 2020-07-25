#!/bin/bash
set -x

function main {
  # default value
  num_inference_batches=500
  batch_size=32
  init_params "$@"
  define_mode
  run_benchmark

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
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;

      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

function define_mode {
    if [[ ${mode} == "accuracy" ]]; then
      mode_cmd=" --accuracy-only"
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --num-inference-batches ${iters} --benchmark True"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_benchmark
function run_benchmark {
    extra_cmd=''
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
    elif [ "${topology}" = "mobileNet1.0" ]; then
        symbol_file=${model_location}'/mobilenet1.0-symbol.json'
        param_file=${model_location}'/mobilenet1.0-0000.params'
        extra_cmd='--rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375'

    elif [ "${topology}" = "mobileNetv2_1.0" ]; then
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
            ${extra_cmd} \
            ${mode_cmd}
}

main "$@"
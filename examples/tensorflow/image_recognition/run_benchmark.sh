#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
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

function define_mode {
    if [[ ${mode} == "accuracy" ]]; then
      mode_cmd=" --benchmark --accuracy-only"
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --steps ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_tuning
function run_benchmark {
    extra_cmd=''
    input="input"
    output="predict"
    image_size=224
    if [ "${topology}" = "resnet50v1.0" ];then
        extra_cmd=' --resize_method crop'
        yaml=resnet50_v1.yaml
    elif [ "${topology}" = "resnet50v1.5" ]; then
        extra_cmd=' --resize_method=crop --r_mean 123.68 --g_mean 116.78 --b_mean 103.94'
        input="input_tensor"
        output="softmax_tensor"
        yaml=resnet50_v1_5.yaml
    elif [ "${topology}" = "resnet101" ]; then
        extra_cmd=' --resize_method vgg --label_adjust'
        output="resnet_v1_101/SpatialSqueeze"
        yaml=resnet101.yaml
    elif [ "${topology}" = "inception_v1" ]; then
        extra_cmd=' --resize_method bilinear'
        output=InceptionV1/Logits/Predictions/Reshape_1
        yaml=inceptionv1.yaml
    elif [ "${topology}" = "inception_v2" ]; then
        extra_cmd=' --resize_method bilinear'
        output=InceptionV2/Predictions/Reshape_1
        yaml=inceptionv2.yaml
    elif [ "${topology}" = "inception_v3" ]; then
        extra_cmd=' --resize_method bilinear'
        image_size=299
        yaml=inceptionv3.yaml
    elif [ "${topology}" = "inception_v4" ]; then
        extra_cmd=' --resize_method bilinear'
        output="InceptionV4/Logits/Predictions"
        image_size=299
        yaml=inceptionv4.yaml
    elif [ "${topology}" = "inception_resnet_v2" ]; then
        extra_cmd="--resize_method bilinear"
        output="InceptionResnetV2/Logits/Predictions"
        image_size=299
        yaml=irv2.yaml
    elif [ "${topology}" = "mobilenetv1" ];then
        extra_cmd=' --resize_method bilinear'
        output="MobilenetV1/Predictions/Reshape_1"
        yaml=mobilenet_v1.yaml
    elif [ "${topology}" = "mobilenetv2" ]; then
        extra_cmd=' --resize_method bilinear'
        output="MobilenetV2/Predictions/Reshape_1"
        yaml=mobilenet_v2.yaml
    elif [ "${topology}" = "mobilenetv3" ]; then
        extra_cmd=' --resize_method bilinear'
        output="MobilenetV3/Predictions/Softmax"
        yaml=mobilenet_v3.yaml
    fi

    python main.py \
            --input-graph ${input_model} \
            --image_size ${image_size} \
            -e 1 -a 28 \
            --input ${input} \
            --output ${output} \
            --data-location ${dataset_location}\
            --config ${yaml} \
            --batch-size ${batch_size} \
            ${extra_cmd} \
            ${mode_cmd}

}

main "$@"
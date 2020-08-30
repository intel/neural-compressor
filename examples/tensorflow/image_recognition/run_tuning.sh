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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
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
    elif [ "${topology}" = "resnetv2_50" ]; then
        extra_cmd=' --resize_method bilinear'
        output="resnet_v2_50/predictions/Reshape_1"
        yaml=resnet_v2_50.yaml
    elif [ "${topology}" = "resnetv2_101" ]; then
        extra_cmd=' --resize_method bilinear'
        output="resnet_v2_101/predictions/Reshape_1"
        yaml=resnet_v2_101.yaml
    elif [ "${topology}" = "resnetv2_152" ]; then
        extra_cmd=' --resize_method bilinear'
        output="resnet_v2_152/predictions/Reshape_1"
        yaml=resnet_v2_152.yaml
    elif [ "${topology}" = "vgg16" ]; then
        extra_cmd=' --resize_method vgg --label_adjust'
        output="vgg_16/fc8/squeezed"
        yaml=vgg16.yaml
    elif [ "${topology}" = "vgg19" ]; then
        extra_cmd=' --resize_method vgg --label_adjust'
        output="vgg_19/fc8/squeezed"
        yaml=vgg19.yaml
    elif [ "${topology}" = "densenet121" ]; then
        extra_cmd=' --resize_method vgg --scale 0.017 --label_adjust'
        input="Placeholder"
        output="densenet121/predictions/Reshape_1"
        yaml=densenet121.yaml
    elif [ "${topology}" = "densenet161" ]; then
        extra_cmd=' --resize_method vgg --scale 0.017 --label_adjust'
        input="Placeholder"
        output="densenet161/predictions/Reshape_1"
        yaml=densenet161.yaml
    elif [ "${topology}" = "densenet169" ]; then
        extra_cmd=' --resize_method vgg --scale 0.017 --label_adjust'
        input="Placeholder"
        output="densenet169/predictions/Reshape_1"
        yaml=densenet169.yaml
    fi

    python main.py \
            --input-graph ${input_model} \
            --output-graph ${output_model} \
            --image_size ${image_size} \
            -e 1 -a 28 \
            --input ${input} \
            --output ${output} \
            --data-location ${dataset_location}\
            --config ${yaml} \
            --batch-size 10 \
            ${extra_cmd} \
            --tune
            

}

main "$@"

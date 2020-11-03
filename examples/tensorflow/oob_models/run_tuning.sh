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

models_need_name=(
efficientnet-b0
efficientnet-b0_auto_aug
efficientnet-b5
efficientnet-b7_auto_aug
vggvox
aipg-vdcnn
arttrack-coco-multi
arttrack-mpii-single
deepvariant_wgs
east_resnet_v1_50
facenet-20180408-102900
handwritten-score-recognition-0003
license-plate-recognition-barrier-0007
optical_character_recognition-text_recognition-tf
PRNet
Resnetv2_200
text-recognition-0012
Hierarchical_LSTM
icnet-camvid-ava-0001
icnet-camvid-ava-sparse-30-0001
icnet-camvid-ava-sparse-60-0001
)

# run_tuning
function run_tuning {
    extra_cmd=''
    input="input"
    output="predict"
    yaml='./config.yaml'
    extra_cmd='--num_warmup 10 -n 500'

    if [[ "${models_need_name[@]}"  =~ "${topology}" ]]; then
      echo "$topology need model name!"
      extra_cmd='--num_warmup 10 -n 500 --model_name '${topology}
    fi

    python tf_benchmark.py \
            --model_path ${input_model} \
            --output_path ${output_model} \
            --yaml ${yaml} \
            ${extra_cmd} \
            --tune

}

main "$@"

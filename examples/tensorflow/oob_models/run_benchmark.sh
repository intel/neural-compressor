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
      echo "For TF OOB models, there is only benchmark mode!, num iter is: ${iters}"
      exit 1
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --num_iter ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
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
Hierarchical
)

models_need_disable_optimize=(
efficientnet-b0
efficientnet-b0_auto_aug
efficientnet-b5
efficientnet-b7_auto_aug
vggvox
)

# run_tuning
function run_benchmark {
    extra_cmd='--num_warmup 10'
    if [[ "${models_need_name[@]}"  =~ "${topology}" ]]; then
      echo "$topology need model name!"
      extra_cmd='--num_warmup 10 --model_name '${topology}
    fi

    if [[ "${models_need_disable_optimize[@]}"  =~ "${topology}" ]]; then
      echo "$topology need model name!"
      extra_cmd='--num_warmup 10 --disable_optimize --model_name '${topology}
    fi

    python tf_benchmark.py \
            --model_path ${input_model} \
            --num_iter ${iters} \
            ${extra_cmd} \
            ${mode_cmd}
}

main "$@"

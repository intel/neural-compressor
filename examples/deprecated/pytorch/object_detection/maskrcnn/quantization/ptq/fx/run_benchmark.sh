#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  tuned_checkpoint=saved_results
  batch_size=1
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
    if [ ! -d 'pytorch/datasets/coco' ]; then
        mkdir -p pytorch/datasets/coco
        ln -s ${dataset_location}/annotations pytorch/datasets/coco/annotations
        ln -s ${dataset_location}/train2017 pytorch/datasets/coco/train2017
        ln -s ${dataset_location}/val2017 pytorch/datasets/coco/val2017
    fi

    if [ ! -f "pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth" ]; then
        ln -s ${input_model} pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth
    fi

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd="--accuracy "
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd="--performance --iter ${iters}"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [ -n "$tuned_checkpoint" ];then
        extra_cmd=$extra_cmd"--tuned_checkpoint ${tuned_checkpoint} "
    fi
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd"--int8"
    fi

    pushd pytorch

    time python tools/test_net.py \
       ${mode_cmd} \
       ${extra_cmd} \
       --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH ${batch_size} SOLVER.MAX_ITER 720000 \
       SOLVER.STEPS "(480000, 640000)" MODEL.DEVICE "cpu"

    popd
}

main "$@"

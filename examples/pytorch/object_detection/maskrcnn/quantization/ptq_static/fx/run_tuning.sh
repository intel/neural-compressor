#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
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
       --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
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
    SCRIPTS=tools/test_net.py

    if [ ! -d 'pytorch/datasets/coco' ]; then
        mkdir -p pytorch/datasets/coco
        ln -s ${dataset_location}/annotations pytorch/datasets/coco/annotations
        ln -s ${dataset_location}/train2017 pytorch/datasets/coco/train2017
        ln -s ${dataset_location}/val2017 pytorch/datasets/coco/val2017
    fi

    if [ ! -f "pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth" ]; then
        ln -s ${input_model} pytorch/e2e_mask_rcnn_R_50_FPN_1x.pth
    fi

    if [ "${SCRIPTS}" = "tools/test_net.py" ];then
        pushd pytorch
        time python $SCRIPTS \
            --tune \
            --tuned_checkpoint $tuned_checkpoint \
            --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
              SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" \
              MODEL.DEVICE "cpu" \
            ${extra_cmd}
        popd
    fi

}

main "$@"

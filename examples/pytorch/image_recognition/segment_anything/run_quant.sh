#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  output_model=saved_results
  pretrained_weight_location=./sam_vit_b_01ec64.pth
  voc_dataset_location=./voc_dataset/VOCdevkit/VOC2012/
  for var in "$@"
  do
    case $var in
      --voc_dataset_location=*)
          voc_dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --pretrained_weight_location=*)
          pretrained_weight_location=$(echo $var |cut -f2 -d=)
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
    python main.py \
            --pretrained_weight_location ${pretrained_weight_location} \
            --tuned_checkpoint ${output_model} \
            --voc_dataset ${voc_dataset_location} \
            --tune True
}

main "$@"
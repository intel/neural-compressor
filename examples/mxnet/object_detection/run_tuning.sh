#!/bin/bash
# set -x

help()
{
   cat <<- EOF

   Desc: Run lpot MXNet Object Detection example.

   -h --help              help info

   --topology             model used for Object Detection, mobilenet1.0 or resnet50_v1, default is mobilenet1.0.

   --dataset_name         coco or voc, default is voc

   --dataset_location     location of dataset

   --input_model          prefix of fp32 model (eg: ./model/ssd-mobilenet )

   --output_model         Best tuning model by lpot will saved in this name prefix. default is './lpot_ssd_model'

EOF
   exit 0
}


function main {

  init_params "$@"
  run_tuning

}

topology='ssd_mobilenet1.0'
dataset='voc'
output_model='./lpot_ssd_model'
dataset_location='~/.mxnet/datasets/'
# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
          topology=${topology: 4}
      ;;
      --dataset_name=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      -h|--help) help
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
    extra_cmd='--data-shape=512 --batch-size=32 --tune'
    python -u eval_ssd.py --network=${topology} \
                          --dataset=${dataset} \
                          --dataset-location=${dataset_location} \
                          --output-graph=${output_model} \
                          ${extra_cmd} 
}

main "$@"

#!/bin/bash
# set -x

help()
{
   cat <<- EOF

   Desc: Run lpot MXNet Object Detection example.

   -h --help              help info

   --topology             model used for Object Detection, mobilenet1.0 or resnet50_v1, default is mobilenet1.0.

   --dataset_name         coco or voc, default is coco

   --dataset_location     location of dataset

   --input_model          (optional) path of the model to benchmark. If not provide, will download at the first running

   --mode                accuracy-only or benchmark
      
   --batch_size

   --iters

EOF
   exit 0
}


function main {
  # default parameters
  topology='mobilenet1.0'
  dataset='voc'
  output_model='./lpot_ssd_model'
  dataset_location='~/.mxnet/datasets/'
  batch_size=32
  iters=10

  init_params "$@"
  define_mode
  run_tuning

}

function define_mode {
    if [[ ${mode} == "accuracy" ]]; then
      mode_cmd=" --accuracy-only"
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --num-iterations ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}


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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
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
    python -u eval_ssd.py --network=${topology} \
                          --dataset=${dataset} \
                          --dataset-location=${dataset_location} \
                          --input-model=${input_model} \
                          --batch-size=${batch_size} \
                          --data-shape=512 \
                          ${mode_cmd}
}

main "$@"

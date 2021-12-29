#!/bin/bash
set -x

function main {
  output_model="./nc_workspace/pytorch/blendcnn/"
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
    esac
  done
  
  if  [ ! "$input_model" ] ;then
    echo "input_model valid, please give right input_model!"
    exit 1
  fi

  if  [ ! "$dataset_location" ] ;then
    echo "dataset_location valid, please give right dataset_location!"
    exit 1
  fi

  if [ ! -x "./models" ]; then
    mkdir "./models"
  fi

  if [ ! -x "./MRPC" ]; then
    mkdir "./MRPC"
  fi

  cp -r ${input_model}/* ./models
  cp -r ${dataset_location}/* ./MRPC
}

# run_tuning
function run_tuning {

    python classify.py --tune \
                       --tuned_checkpoint ${tuned_checkpoint} \
                       --input_model=${input_model}/model_final.pt \
                       --dataset_location=${dataset_location}/dev.tsv
}

main "$@"

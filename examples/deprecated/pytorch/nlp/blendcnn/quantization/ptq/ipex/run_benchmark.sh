#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=32
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
  
  if [ ! -x "./models" ]; then
    mkdir "./models"
  fi

  if [ ! -x "./MRPC" ]; then
    mkdir "./MRPC"
  fi

  if  [ ! "$input_model" ] ;then
    echo "input_model valid, please give right input_model!"
  else
    cp -r ${input_model}/* ./models
  fi

  if  [ ! "$dataset_location" ] ;then
    echo "dataset_location valid, please give right dataset_location!"
  else
    cp -r ${dataset_location}/* ./MRPC
  fi
  
}


# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy"
    elif [[ ${mode} == "performance" ]]; then
        mode_cmd=" --iter ${iters} --performance  --batch_size ${batch_size}"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd="--int8 "
    else
        extra_cmd="--input_model ${input_model}/model_final.pt"
    fi

    python classify.py \
            --tuned_checkpoint ${tuned_checkpoint} \
            ${mode_cmd} \
            ${extra_cmd}
}

main "$@"

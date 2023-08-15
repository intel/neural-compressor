#!/bin/bash
set -x

function main {

  init_params "$@"
  run_fine_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --train_dataset_location=*)
          train_dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_fine_tuning
function run_fine_tuning {

    python main.py \
            --model_name_or_path microsoft/codebert-base \
            --train_data_path ${train_dataset_location} \
            --data_path ${dataset_location} \
            --fine_tune

}

main "$@"

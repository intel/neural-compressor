#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning
}

# init params
function init_params {
  batch_size=1
  output_model=saved_results
  folder_in_archive=LibriSpeech
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
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

  url=${dataset_location##*/}
  root=${dataset_location%${folder_in_archive}*}
}

# run_tuning
function run_tuning {
    
    python run_asr.py \
            --model $input_model \
            --root $root \
            --folder_in_archive $folder_in_archive \
            --url $url \
            --tune \
            --batch_size $batch_size \
            --tuned_checkpoint ${output_model}

}

main "$@"

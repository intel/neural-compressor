#!/bin/bash
set -x

function main {

  init_params "$@"
  prepare_dataset

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --download_dir=*)
          download_dir=$(echo $var |cut -f2 -d=)
      ;;
      --convert_dir=*)
          convert_dir=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

  mkdir -p $download_dir $convert_dir
}

# prepare_dataset
function prepare_dataset {
  # if you already have origin dataset, set stage=2, make sure to extract it \
  # and change the origin dataset path to your path
  stage=1
  
  # Download dataset
  if [[ $stage -le 1 ]]; then
    python pytorch/utils/download_librispeech.py \
            pytorch/utils/librispeech-inference.csv \
            $download_dir \
            -e $download_dir
  fi

  # Convert dataset
  if [[ $stage -le 2 ]]; then
    python pytorch/utils/convert_librispeech.py \
        --input_dir $download_dir/LibriSpeech/dev-clean \
        --dest_dir $convert_dir/dev-clean-wav \
        --output_json $convert_dir/dev-clean-wav.json
  fi
}

main "$@"
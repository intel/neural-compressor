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
      --origin_dir=*)
          origin_dir=$(echo $var |cut -f2 -d=)
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
  python upscale_coco/upscale_coco.py --inputs $origin_dir \
  --outputs $convert_dir --size 1200 1200 --format png
}

main "$@"
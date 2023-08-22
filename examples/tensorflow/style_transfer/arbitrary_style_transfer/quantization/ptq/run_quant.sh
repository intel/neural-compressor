#!/bin/bash
# set -x

function main {

  init_params "$@"

  run_tuning

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo "$var" |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    style_images=$(echo ${dataset_location} | awk -F ',' '{print $1}')
    content_images=$(echo ${dataset_location} | awk -F ',' '{print $2}')
    echo "$style_images, $content_images"

    python style_tune.py \
            --input_model "${input_model}" \
            --style_images_paths "${style_images}" \
            --content_images_paths "${content_images}" \
            --config "./conf.yaml" \
            --tune=True \
            --output_model "${output_model}"
}

main "$@"

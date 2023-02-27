#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_tuning
function run_benchmark {
    style_images=$(echo ${dataset_location} | awk -F ',' '{print $1}')
    content_images=$(echo ${dataset_location} | awk -F ',' '{print $2}')
    echo "$style_images, $content_images"

    python style_tune.py \
            --input_model "${input_model}" \
            --style_images_paths "${style_images}" \
            --content_images_paths "${content_images}" \
            --batch_size "${batch_size}" \
            --tune=False \
            --output_model "${output_model}"

}

main "$@"

#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --model_name=*)
          model_name=$(echo $var |cut -f2 -d=)
      ;;
      --image_folder=*)
          image_folder=$(echo $var |cut -f2 -d=)
      ;;
      --question_file=*)
          question_file=$(echo $var |cut -f2 -d=)
      ;;
      --output_dir=*)
          output_dir=$(echo $var |cut -f2 -d=)
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
    python main.py \
            --model_name ${model_name} \
            --nsamples 512 \
            --quantize \
            --image_folder ${image_folder} \
            --question_file ${question_file} \
            --output_dir ${output_dir}
}

main "$@"

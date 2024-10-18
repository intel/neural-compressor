#!/bin/bash
set -x

function main {

  init_params "$@"
  run_evaluation

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --model_name=*)
          model_name=$(echo $var |cut -f2 -d=)
      ;;
      --eval_question_file=*)
          eval_question_file=$(echo $var |cut -f2 -d=)
      ;;
      --eval_image_folder=*)
          eval_image_folder=$(echo $var |cut -f2 -d=)
      ;;
      --eval_annotation_file=*)
          eval_annotation_file=$(echo $var |cut -f2 -d=)
      ;; 
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_evaluation
function run_evaluation {
    python main.py \
            --accuracy \
            --model_name ${model_name} \
            --eval_question_file ${eval_question_file} \
            --eval_image_folder ${eval_image_folder} \
            --eval_annotation_file ${eval_annotation_file}
}

main "$@"


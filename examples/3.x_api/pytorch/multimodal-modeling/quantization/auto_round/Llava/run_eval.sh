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
      --eval-question-file=*)
          eval-question-file=$(echo $var |cut -f2 -d=)
      ;;
      --eval-image-folder=*)
          eval-image-folder=$(echo $var |cut -f2 -d=)
      ;;
      --eval-annotation-file=*)
          eval-annotation-file=$(echo $var |cut -f2 -d=)
      ;;
      --eval-result-file=*)
          eval-result-file=$(echo $var |cut -f2 -d=)
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
    python mm_evaluation/textvqa.py \
            --model_name ${model_name} \
            --eval-question-file ${eval-question-file} \
            --eval-image-folder ${eval-image-folder} \
            --eval-annotation-file ${eval-annotation-file} \
            --eval-result-file ${eval-result-file} \
            --trust_remote_code
}

main "$@"

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
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_evaluation
function run_evaluation {
    python mm_evaluation/main.py \
            --model_name ${model_name} \
            --trust_remote_code \
            --eval_bs 4
}

main "$@"

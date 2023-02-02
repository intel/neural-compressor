#!/bin/bash
set -x

function main {
  init_params "$@"
  run_export

}

# init params
function init_params {

  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_export
function run_export {
    python main.py \
            --input-graph ${input_model} \
            --output-graph ${output_model} \
            --export
}

main "$@"

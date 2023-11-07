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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --sq=*)
          sq=$(echo ${var} |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    ext_cmd=""
    if [[ ${sq} == "True" ]]; then
        ext_cmd="--sq"
    fi
    python main.py \
        --model_name_or_path ${input_model} \
        ${ext_cmd}
}

main "$@"
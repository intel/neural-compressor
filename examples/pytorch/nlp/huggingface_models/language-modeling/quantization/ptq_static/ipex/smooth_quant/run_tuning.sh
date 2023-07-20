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
    esac
  done

}

# run_tuning
function run_tuning {

    ext_cmd=""
    if [[ ${input_model} == *"gpt-j-6B" ]]; then
        ext_cmd="--alpha auto --fallback_add"
    fi

    python eval_lambada.py \
        --model_name_or_path ${input_model} \
        --int8 \
        --sq \
        ${ext_cmd}
}

main "$@"

#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
        topology=$(echo $var | cut -f2 -d=)
      ;;
      --input_model=*)
        input_model=$(echo $var | cut -f2 -d=)
      ;;
      --task=*)
        task=$(echo $var | cut -f2 -d=)
      ;;
      --output_model=*)
        tuned_checkpoint=$(echo $var | cut -f2 -d=)
      ;;
      *)
        echo "Error: No such parameter: ${var}"
        exit 1
      ;;
    esac
  done
}

function run_tuning {
  tuned_checkpoint=${tuned_checkpoint:="./tmp_autoround"}
  task=${task:="t2v"}

  if [ "${topology}" = "wan_fp8" ]; then
    extra_cmd="--scheme FP8"
  elif [ "${topology}" = "wan_mxfp8" ]; then
    extra_cmd="--scheme MXFP8"
  else
    echo "Error: unsupported topology ${topology}, use wan_fp8 or wan_mxfp8"
    exit 1
  fi

  python3 main.py \
    --model ${input_model} \
    --task ${task} \
    --output_dir ${tuned_checkpoint} \
    --quantize \
    ${extra_cmd}
}

main "$@"

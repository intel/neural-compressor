#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

function ensure_wan_repo {
  if [ -d "${wan_dir}" ]; then
    return
  fi

  echo "Error: Wan2.2 directory not found: ${wan_dir}"
  echo "Please prepare Wan2.2 manually and pass --wan_dir=/path/to/Wan2.2 if needed."
  exit 1
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
      --wan_dir=*)
        wan_dir=$(echo $var | cut -f2 -d=)
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
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  wan_dir=${wan_dir:="${script_dir}/Wan2.2"}

  if [ "${topology}" = "wan_fp8" ]; then
    scheme="FP8"
  elif [ "${topology}" = "wan_mxfp8" ]; then
    scheme="MXFP8"
  else
    echo "Error: unsupported topology ${topology}, use wan_fp8 or wan_mxfp8"
    exit 1
  fi

  if [ "${task}" = "s2v" ]; then
    ensure_wan_repo
    env "PYTHONPATH=${wan_dir}:${PYTHONPATH}" python3 wan_s2v.py \
      --model ${input_model} \
      --task s2v-14B \
      --scheme ${scheme} \
      --quantize \
      --output_dir ${tuned_checkpoint}
  else
    python3 main.py \
      --model ${input_model} \
      --task ${task} \
      --scheme ${scheme} \
      --quantize \
      --output_dir ${tuned_checkpoint}
  fi
}

main "$@"

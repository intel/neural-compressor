#!/bin/bash
set -x

function main {
  init_params "$@"
  run_benchmark
}

function ensure_vbench_repo {
  if [ ! -d "VBench" ]; then
    echo "VBench directory not found. Start cloning https://github.com/Vchitect/VBench.git ..."
    git clone https://github.com/Vchitect/VBench.git
    if [ $? -ne 0 ]; then
      echo "Error: failed to clone VBench."
      exit 1
    fi
  fi
}

function prepare_vbench_inputs {
  if [ "${task}" = "t2v" ] && [ -z "${prompt_file}" ]; then
    echo "Error: --prompt_file is required for task=t2v"
    exit 1
  fi

  if [ "${task}" = "i2v" ]; then
    if [ -z "${image_folder}" ]; then
      echo "Error: --image_folder is required for task=i2v"
      exit 1
    fi
    if [ -z "${info_json}" ]; then
      echo "Error: --info_json is required for task=i2v"
      exit 1
    fi
  fi

  if [ -n "${prompt_file}" ] && [ ! -f "${prompt_file}" ]; then
    echo "Error: prompt_file not found: ${prompt_file}"
    exit 1
  fi
  if [ -n "${image_folder}" ] && [ ! -d "${image_folder}" ]; then
    echo "Error: image_folder not found: ${image_folder}"
    exit 1
  fi
  if [ -n "${info_json}" ] && [ ! -f "${info_json}" ]; then
    echo "Error: info_json not found: ${info_json}"
    exit 1
  fi
}

function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
        topology="${var#*=}"
      ;;
      --input_model=*)
        input_model="${var#*=}"
      ;;
      --task=*)
        task="${var#*=}"
      ;;
      --quantized_model=*)
        tuned_checkpoint="${var#*=}"
      ;;
      --output_video_path=*)
        output_video_path="${var#*=}"
      ;;
      --prompt_file=*)
        prompt_file="${var#*=}"
      ;;
      --image_folder=*)
        image_folder="${var#*=}"
      ;;
      --info_json=*)
        info_json="${var#*=}"
      ;;
      --limit=*)
        limit="${var#*=}"
      ;;
      --accuracy)
        accuracy=true
      ;;
      *)
        echo "Error: No such parameter: ${var}"
        exit 1
      ;;
    esac
  done
}

function run_benchmark {
  task=${task:="t2v"}
  limit=${limit:=-1}
  tuned_checkpoint=${tuned_checkpoint:="./tmp_autoround"}
  output_video_path=${output_video_path:="./tmp_video"}
  accuracy=${accuracy:=false}

  if [[ ! "${output_video_path}" = /* ]]; then
    output_video_path=$(realpath -s "$(pwd)/${output_video_path}")
  fi

  if [ "${topology}" = "wan_bf16" ]; then
    scheme="BF16"
  elif [ "${topology}" = "wan_fp8" ]; then
    scheme="FP8"
  elif [ "${topology}" = "wan_mxfp8" ]; then
    scheme="MXFP8"
  else
    echo "Error: unsupported topology ${topology}, use wan_bf16/wan_fp8/wan_mxfp8"
    exit 1
  fi

  ensure_vbench_repo

  prepare_vbench_inputs

  benchmark_cmd=(
    python3 main.py
    --model "${input_model}"
    --task "${task}"
    --scheme "${scheme}"
    --output_dir "${tuned_checkpoint}"
    --output_video_path "${output_video_path}"
    --limit "${limit}"
    --inference
  )

  if [ -n "${prompt_file}" ]; then
    benchmark_cmd+=(--prompt_file "${prompt_file}")
  fi
  if [ -n "${image_folder}" ]; then
    benchmark_cmd+=(--image_folder "${image_folder}")
  fi
  if [ -n "${info_json}" ]; then
    benchmark_cmd+=(--info_json "${info_json}")
  fi

  "${benchmark_cmd[@]}"

  if [ "${accuracy}" = "true" ]; then
    if [ "${task}" = "t2v" ]; then
      echo "Start VBench evaluation for t2v..."
      pushd VBench
      python evaluate.py \
        --dimension subject_consistency motion_smoothness aesthetic_quality imaging_quality overall_consistency \
        --videos_path "${output_video_path}" \
        --mode=vbench_standard
      popd
    elif [ "${task}" = "i2v" ]; then
      echo "Start VBench evaluation for i2v..."
      pushd VBench
      python evaluate_i2v.py \
        --dimension i2v_background i2v_subject subject_consistency background_consistency motion_smoothness \
        --videos_path "${output_video_path}" \
        --mode=vbench_standard
      popd
    else
      echo "--accuracy does not support task=${task}. Supported tasks: t2v, i2v."
      exit 1
    fi
  else
    echo "Video generation finished. Use --accuracy to run VBench evaluation for t2v/i2v."
  fi
}

main "$@"

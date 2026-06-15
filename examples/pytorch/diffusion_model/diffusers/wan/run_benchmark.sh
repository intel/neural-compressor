#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

function main {
  init_params "$@"
  run_benchmark
}

function ensure_vbench_repo {
  if [ ! -d "${vbench_dir}" ]; then
    echo "VBench directory not found. Start cloning https://github.com/Vchitect/VBench.git ..."
    git clone https://github.com/Vchitect/VBench.git "${vbench_dir}"
    if [ $? -ne 0 ]; then
      echo "Error: failed to clone VBench."
      exit 1
    fi
  fi
}

function prepare_vbench_inputs {
  if [ "${task}" = "t2v" ]; then
    if [ -z "${prompt_folder}" ]; then
      echo "Error: --prompt_folder is required for task=t2v"
      exit 1
    fi
    if [ -z "${dimension}" ]; then
      echo "Error: --dimension is required for task=t2v"
      exit 1
    fi
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
    if [ -z "${dimension}" ]; then
      echo "Error: --dimension is required for task=i2v"
      exit 1
    fi
  fi

  if [ -n "${prompt_folder}" ] && [ ! -d "${prompt_folder}" ]; then
    echo "Error: prompt_folder not found: ${prompt_folder}"
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
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --topology=*)
        topology="${1#*=}"
        shift
      ;;
      --topology)
        topology="$2"
        shift 2
      ;;
      --input_model=*)
        input_model="${1#*=}"
        shift
      ;;
      --input_model)
        input_model="$2"
        shift 2
      ;;
      --task=*)
        task="${1#*=}"
        shift
      ;;
      --task)
        task="$2"
        shift 2
      ;;
      --quantized_model=*)
        tuned_checkpoint="${1#*=}"
        shift
      ;;
      --quantized_model)
        tuned_checkpoint="$2"
        shift 2
      ;;
      --output_video_path=*)
        output_video_path="${1#*=}"
        shift
      ;;
      --output_video_path)
        output_video_path="$2"
        shift 2
      ;;
      --prompt_folder=*)
        prompt_folder="${1#*=}"
        shift
      ;;
      --prompt_folder)
        prompt_folder="$2"
        shift 2
      ;;
      --image_folder=*)
        image_folder="${1#*=}"
        shift
      ;;
      --image_folder)
        image_folder="$2"
        shift 2
      ;;
      --info_json=*)
        info_json="${1#*=}"
        shift
      ;;
      --info_json)
        info_json="$2"
        shift 2
      ;;
      --dimension=*)
        dimension="${1#*=}"
        shift
      ;;
      --dimension)
        dimension="$2"
        shift 2
      ;;
      --gpu_ids=*)
        gpu_ids="${1#*=}"
        shift
      ;;
      --gpu_ids)
        gpu_ids="$2"
        shift 2
      ;;
      --limit=*)
        limit="${1#*=}"
        shift
      ;;
      --limit)
        limit="$2"
        shift 2
      ;;
      --mxfp8_chunk_rows=*)
        mxfp8_chunk_rows="${1#*=}"
        shift
      ;;
      --mxfp8_chunk_rows)
        mxfp8_chunk_rows="$2"
        shift 2
      ;;
      --disable_mxfp8_inplace_qdq)
        disable_mxfp8_inplace_qdq=true
        shift
      ;;
      --accuracy)
        accuracy=true
        shift
      ;;
      --vbench_dir=*)
        vbench_dir="${1#*=}"
        shift
      ;;
      --vbench_dir)
        vbench_dir="$2"
        shift 2
      ;;
      *)
        echo "Error: No such parameter: $1"
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
  disable_mxfp8_inplace_qdq=${disable_mxfp8_inplace_qdq:=false}
  vbench_dir=${vbench_dir:="${SCRIPT_DIR}/VBench"}

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

  normalized_dimensions="${dimension//,/ }"
  read -r -a dimension_list <<< "${normalized_dimensions}"

  if [ -n "${gpu_ids}" ]; then
    gpu_list="${gpu_ids}"
  else
    gpu_list="${CUDA_VISIBLE_DEVICES:-}"
  fi

  if [ -n "${gpu_list}" ]; then
    normalized_gpu_ids="${gpu_list//,/ }"
    read -r -a gpu_array <<< "${normalized_gpu_ids}"
    visible_gpus=${#gpu_array[@]}
    echo "visible_gpus: ${visible_gpus}"
  else
    gpu_array=()
  fi

  mkdir -p "${output_video_path}"
  shard_tmp_root="${output_video_path}/.prompt_shards"

  function build_benchmark_cmd {
    local cur_prompt_folder="$2"
    local cur_info_json="$3"
    local cmd=(
      python3 main.py
      --model "${input_model}"
      --task "${task}"
      --scheme "${scheme}"
      --output_dir "${tuned_checkpoint}"
      --output_video_path "${output_video_path}"
      --limit "${limit}"
      --inference
    )

    if [ -n "${cur_prompt_folder}" ]; then
      cmd+=(--prompt_folder "${cur_prompt_folder}")
    elif [ -n "${prompt_folder}" ]; then
      cmd+=(--prompt_folder "${prompt_folder}")
    fi
    if [ -n "${image_folder}" ]; then
      cmd+=(--image_folder "${image_folder}")
    fi
    if [ -n "${cur_info_json}" ]; then
      cmd+=(--info_json "${cur_info_json}")
    elif [ -n "${info_json}" ]; then
      cmd+=(--info_json "${info_json}")
    fi
    if [ -n "$1" ]; then
      cmd+=(--dimension "$1")
    fi
    if [ -n "${mxfp8_chunk_rows}" ]; then
      cmd+=(--mxfp8_chunk_rows "${mxfp8_chunk_rows}")
    fi
    if [ "${disable_mxfp8_inplace_qdq}" = "true" ]; then
      cmd+=(--disable_mxfp8_inplace_qdq)
    fi

    printf '%q ' "${cmd[@]}"
  }

  if [ ${#gpu_array[@]} -eq 0 ]; then
    if [ ${#dimension_list[@]} -eq 0 ]; then
      eval "$(build_benchmark_cmd "" "" "")"
    else
      for cur_dimension in "${dimension_list[@]}"; do
        eval "$(build_benchmark_cmd "${cur_dimension}" "" "")"
      done
    fi
  else
    if [ ${#dimension_list[@]} -eq 0 ]; then
      echo "Error: multi-GPU sharding requires --dimension"
      exit 1
    fi

    num_shards=${#gpu_array[@]}
    for cur_dimension in "${dimension_list[@]}"; do
      dim_shard_root="${shard_tmp_root}/${cur_dimension}"
      rm -rf "${dim_shard_root}"
      if [ "${task}" = "t2v" ]; then
        prompt_file="${prompt_folder}/${cur_dimension}.txt"
        python3 split_t2v_prompts.py \
          --prompt_file "${prompt_file}" \
          --num_shards "${num_shards}" \
          --output_root "${dim_shard_root}"
      else
        python3 split_i2v_info.py \
          --info_json "${info_json}" \
          --dimension "${cur_dimension}" \
          --num_shards "${num_shards}" \
          --output_root "${dim_shard_root}"
      fi

      program_pid=()
      for shard_id in "${!gpu_array[@]}"; do
        gpu_id="${gpu_array[$shard_id]}"
        log_suffix="${cur_dimension}"
        if [ -z "${log_suffix}" ]; then
          log_suffix="all"
        fi
        log_file="${output_video_path}/${log_suffix}.gpu${gpu_id}.log"
        shard_prompt_folder=""
        shard_info_json=""

        if [ "${task}" = "t2v" ]; then
          shard_prompt_folder="${dim_shard_root}/shard_${shard_id}"
        else
          shard_info_json="${dim_shard_root}/shard_${shard_id}/info.json"
        fi

        cmd="$(build_benchmark_cmd "${cur_dimension}" "${shard_prompt_folder}" "${shard_info_json}")"
        CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "${cmd}" > "${log_file}" 2>&1 &
        program_pid+=("$!")
        echo "Start (PID: ${program_pid[-1]}, GPU: ${gpu_id}, dimension: ${cur_dimension})"
      done

      for pid in "${program_pid[@]}"; do
        wait "${pid}" || exit 1
      done
    done
  fi

  if [ "${accuracy}" = "true" ]; then
    if [ "${task}" = "t2v" ]; then
      echo "Start VBench evaluation for t2v..."
      pushd "${vbench_dir}"
      python evaluate.py \
        --dimension "subject_consistency motion_smoothness aesthetic_quality imaging_quality overall_consistency" \
        --videos_path "${output_video_path}" \
        --mode=vbench_standard 
      popd
    elif [ "${task}" = "i2v" ]; then
      echo "Start VBench evaluation for i2v..."
      pushd "${vbench_dir}"
      python evaluate_i2v.py \
        --dimension "i2v_background i2v_subject subject_consistency background_consistency motion_smoothness" \
        --videos_path "${output_video_path}" \
        --ratio "16-9" \
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

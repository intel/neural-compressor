#!/bin/bash
set -x

function main {
  init_params "$@"
  run_benchmark
}

function ensure_vbench_repo {
  if [ -d "${vbench_dir}" ]; then
    return
  fi

  echo "Error: VBench directory not found: ${vbench_dir}"
  echo "Please prepare VBench manually and pass --vbench_dir=/path/to/VBench if needed."
  exit 1
}

function ensure_vbench_data {
  local prompt_root="${vbench_dir}/prompts/prompts_per_dimension"
  local i2v_image_root="${vbench_dir}/vbench2_beta_i2v/data/crop/16-9"
  local i2v_info_file="${vbench_dir}/vbench2_beta_i2v/vbench2_i2v_full_info.json"

  if [ -d "${prompt_root}" ] && [ -d "${i2v_image_root}" ] && [ -f "${i2v_info_file}" ]; then
    return
  fi

  echo "Error: VBench data is incomplete under ${vbench_dir}."
  echo "Please prepare VBench data manually (for example run vbench2_beta_i2v/download_data.sh in your VBench repo)."
  exit 1
}

function ensure_wan_repo {
  if [ -d "${wan_dir}" ]; then
    return
  fi

  echo "Error: Wan2.2 directory not found: ${wan_dir}"
  echo "Please prepare Wan2.2 manually and pass --wan_dir=/path/to/Wan2.2 if needed."
  exit 1
}

function ensure_s2v_manifest {
  if [ -z "${manifest_path}" ]; then
    echo "Error: --manifest_path is required for task=s2v"
    exit 1
  fi

  if [ ! -f "${manifest_path}" ]; then
    echo "Error: manifest_path not found: ${manifest_path}"
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
      --manifest_path=*)
        manifest_path="${1#*=}"
        shift
      ;;
      --manifest_path)
        manifest_path="$2"
        shift 2
      ;;
      --wan_dir=*)
        wan_dir="${1#*=}"
        shift
      ;;
      --wan_dir)
        wan_dir="$2"
        shift 2
      ;;
      --vbench_dir=*)
        vbench_dir="${1#*=}"
        shift
      ;;
      --vbench_dir)
        vbench_dir="$2"
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
      --s2v_eval_output=*)
        s2v_eval_output="${1#*=}"
        shift
      ;;
      --s2v_eval_output)
        s2v_eval_output="$2"
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
  s2v_eval_output=${s2v_eval_output:=""}
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  wan_dir=${wan_dir:="${script_dir}/Wan2.2"}
  vbench_dir=${vbench_dir:="${script_dir}/VBench"}

  if [[ ! "${output_video_path}" = /* ]]; then
    output_video_path=$(realpath -s "$(pwd)/${output_video_path}")
  fi

  eval_results_path="${output_video_path}_eval_results"

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

  if [ "${task}" != "s2v" ]; then
    ensure_vbench_repo
    ensure_vbench_data
    if [ "${task}" = "t2v" ]; then
      prompt_folder=${prompt_folder:="${vbench_dir}/prompts/prompts_per_dimension"}
    fi
    if [ "${task}" = "i2v" ]; then
      image_folder=${image_folder:="${vbench_dir}/vbench2_beta_i2v/data/crop/16-9"}
      info_json=${info_json:="${vbench_dir}/vbench2_beta_i2v/vbench2_i2v_full_info.json"}
    fi
  else
    ensure_wan_repo
    ensure_s2v_manifest
  fi

  if [ "${task}" != "s2v" ] && [ -z "${dimension}" ]; then
    echo "Error: --dimension is required for task=${task}"
    exit 1
  fi

  mkdir -p "${output_video_path}"
  mkdir -p "${eval_results_path}"

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

  if [ "${task}" = "s2v" ]; then
    function build_s2v_cmd {
      local cur_manifest_path="$1"
      local cmd=(
        env "PYTHONPATH=${wan_dir}:${PYTHONPATH}" python3 wan_s2v.py
        --model "${input_model}"
        --task "s2v-14B"
        --scheme "${scheme}"
        --output_video_path "${output_video_path}"
        --manifest_path "${cur_manifest_path}"
        --inference
      )

      if [ "${scheme}" != "BF16" ]; then
        cmd+=(--quantized_model "${tuned_checkpoint}")
      fi

      if [ -n "${mxfp8_chunk_rows}" ]; then
        cmd+=(--mxfp8_chunk_rows "${mxfp8_chunk_rows}")
      fi
      if [ "${disable_mxfp8_inplace_qdq}" = "true" ]; then
        cmd+=(--disable_mxfp8_inplace_qdq)
      fi

      printf "%q " "${cmd[@]}"
    }

    if [ ${#gpu_array[@]} -eq 0 ]; then
      run_cmd="$(build_s2v_cmd "${manifest_path}")"
      eval "${run_cmd}"
    else
      num_shards=${#gpu_array[@]}
      s2v_shard_root="${eval_results_path}/.manifest_shards"
      rm -rf "${s2v_shard_root}"

      python3 split_s2v_manifest.py \
        --manifest_path "${manifest_path}" \
        --num_shards "${num_shards}" \
        --output_root "${s2v_shard_root}"

      program_pid=()
      for shard_id in "${!gpu_array[@]}"; do
        gpu_id="${gpu_array[$shard_id]}"
        shard_manifest_path="${s2v_shard_root}/shard_${shard_id}/manifest.json"
        if [ ! -f "${shard_manifest_path}" ]; then
          echo "Skip empty shard_${shard_id} on GPU ${gpu_id}"
          continue
        fi
        log_file="${eval_results_path}/s2v.gpu${gpu_id}.log"
        run_cmd="$(build_s2v_cmd "${shard_manifest_path}")"
        CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "${run_cmd}" > "${log_file}" 2>&1 &
        program_pid+=("$!")
        echo "Start (PID: ${program_pid[-1]}, GPU: ${gpu_id}, shard: ${shard_id})"
      done

      if [ ${#program_pid[@]} -eq 0 ]; then
        echo "Error: no non-empty s2v shards to run. Check --manifest_path content."
        exit 1
      fi

      for pid in "${program_pid[@]}"; do
        wait "${pid}" || exit 1
      done
    fi
  else
    normalized_dimensions="${dimension//,/ }"
    read -r -a dimension_list <<< "${normalized_dimensions}"

    shard_tmp_root="${eval_results_path}/.prompt_shards"

    function build_benchmark_cmd {
      local cur_prompt_folder="$2"
      local cur_info_json="$3"
      local cmd=(
        python3 main.py
        --model "${input_model}"
        --task "${task}"
        --scheme "${scheme}"
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

      printf "%q " "${cmd[@]}"
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
          log_file="${eval_results_path}/${log_suffix}.gpu${gpu_id}.log"
          shard_prompt_folder=""
          shard_info_json=""

          if [ "${task}" = "t2v" ]; then
            shard_prompt_folder="${dim_shard_root}/shard_${shard_id}"
          else
            shard_info_json="${dim_shard_root}/shard_${shard_id}/info.json"
          fi

          run_cmd="$(build_benchmark_cmd "${cur_dimension}" "${shard_prompt_folder}" "${shard_info_json}")"
          CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "${run_cmd}" > "${log_file}" 2>&1 &
          program_pid+=("$!")
          echo "Start (PID: ${program_pid[-1]}, GPU: ${gpu_id}, dimension: ${cur_dimension})"
        done

        for pid in "${program_pid[@]}"; do
          wait "${pid}" || exit 1
        done
      done
    fi
  fi

  if [ "${accuracy}" = "true" ]; then
    if [ "${task}" = "t2v" ]; then
      echo "Start VBench evaluation for t2v..."
      pushd "${vbench_dir}"
        eval_dimension_list=("${dimension_list[@]}")
        for extra_dimension in motion_smoothness aesthetic_quality imaging_quality; do
          if [[ ! " ${eval_dimension_list[*]} " =~ " ${extra_dimension} " ]]; then
            eval_dimension_list+=("${extra_dimension}")
          fi
        done
        python evaluate.py \
          --dimension "${eval_dimension_list[@]}" \
          --videos_path "${output_video_path}" \
            --output_path "${eval_results_path}/vbench_t2v" \
          --mode=vbench_standard
      popd
    elif [ "${task}" = "i2v" ]; then
      echo "Start VBench evaluation for i2v..."
      pushd "${vbench_dir}"
        eval_dimension_list=("${dimension_list[@]}")
        for extra_dimension in subject_consistency background_consistency motion_smoothness; do
          if [[ ! " ${eval_dimension_list[*]} " =~ " ${extra_dimension} " ]]; then
            eval_dimension_list+=("${extra_dimension}")
          fi
        done
        python evaluate_i2v.py \
          --dimension "${eval_dimension_list[@]}" \
          --videos_path "${output_video_path}" \
            --output_path "${eval_results_path}/vbench_i2v" \
          --ratio "16-9" \
          --mode=vbench_standard
      popd
    elif [ "${task}" = "s2v" ]; then
      echo "Start s2v evaluation..."
      s2v_eval_script="${script_dir}/evaluate_manifest_no_gt.py"
      s2v_eval_manifest="${eval_results_path}/s2v_manifest_with_generate_video.json"
      if [ ! -f "${s2v_eval_script}" ]; then
        echo "Error: s2v evaluation script not found: ${s2v_eval_script}"
        exit 1
      fi
      if [ -z "${s2v_eval_output}" ]; then
        s2v_eval_output="${eval_results_path}/evaluation_no_gt_metrics_s2v.json"
      fi

      eval_cmd=(
        python3 "${s2v_eval_script}"
        --manifest "${manifest_path}"
        --generated_video_dir "${output_video_path}"
        --matched_manifest_output "${s2v_eval_manifest}"
        --output "${s2v_eval_output}"
        --max_frames "32"
        --metric_size "192"
      )
      printf "%q " "${eval_cmd[@]}" && echo
      "${eval_cmd[@]}"

      echo "S2V evaluation finished."
      echo "- matched manifest: ${s2v_eval_manifest}"
      echo "- metrics output: ${s2v_eval_output}"
    else
      echo "--accuracy currently does not support task=${task}. Generated videos are saved at ${output_video_path}."
    fi
  else
    if [ "${task}" = "s2v" ]; then
      echo "S2V generation finished. Videos are in ${output_video_path}."
    else
      echo "Video generation finished. Use --accuracy to run VBench evaluation for t2v/i2v."
    fi
  fi
}

main "$@"

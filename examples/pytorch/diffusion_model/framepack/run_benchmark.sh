#!/bin/bash
set -ex

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --ratio=*)
          ratio=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --limit=*)
          limit=$(echo $var |cut -f2 -d=)
      ;;
      --output_video_path=*)
          output_video_path=$(echo $var |cut -f2 -d=)
      ;;
      --result_path=*)
          result_path=$(echo $var |cut -f2 -d=)
      ;;
      --dimension_list=*)
          dimension_list=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    limit=${limit:=-1}
    ratio=${ratio:="16-9"}
    output_video_path=${output_video_path:="./tmp_videos"}
    result_path=${result_path:="./eval_result"}

    if [[ ! "${result_path}" = /* ]]; then
        result_path=$(realpath -s "$(pwd)/$result_path")
    fi

    if [[ ! "${output_video_path}" = /* ]]; then
        output_video_path=$(realpath -s "$(pwd)/$output_video_path")
    fi

    if [ "${topology}" = "FP8" ]; then
        extra_cmd="--scheme FP8 --quantize --inference"
    elif [ "${topology}" = "MXFP8" ]; then
        extra_cmd="--scheme MXFP8 --quantize --inference"
    elif [ "${topology}" = "BF16" ]; then
        extra_cmd="--inference"
    fi

    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        gpu_list="${CUDA_VISIBLE_DEVICES:-}"
        IFS=',' read -ra gpu_ids <<< "$gpu_list"
        visible_gpus=${#gpu_ids[@]}
        echo "visible_gpus: ${visible_gpus}"

        torchrun --nproc_per_node=${visible_gpus} main.py \
                --output_video_path ${output_video_path} \
                --dataset_location ${dataset_location} \
                --limit ${limit} \
                --ratio ${ratio} \
                --dimension_list ${dimension_list} \
                ${extra_cmd}
    else
        python3 main.py \
            --output_video_path ${output_video_path} \
            --dataset_location ${dataset_location} \
            --limit ${limit} \
            --ratio ${ratio} \
            --dimension_list ${dimension_list} \
            ${extra_cmd}
    fi

    echo "Start calculating final score..."
    cd ${dataset_location}
    python evaluate_i2v.py \
        --videos_path ${output_video_path} \
        --dimension ${dimension_list} \
        --output_path ${result_path} \
        --ratio ${ratio} 2>&1 | tee output.log
    result_file=$(echo output.log | grep -i "Evaluation results saved to " | awk '{print $NF}')

    echo "Evaluation results saved to ${result_file}"
    zip -r "${result_path}.zip" ${result_path}
    python scripts/cal_i2v_final_score.py --zip_file "${result_path}.zip" --model_name "framepack"

}

main "$@"


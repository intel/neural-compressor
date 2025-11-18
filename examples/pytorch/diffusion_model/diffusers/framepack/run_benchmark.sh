#!/bin/bash
set -x

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

        IFS=' ' read -ra dimensions <<< "$dimension_list"
        dimension_num=${#dimensions[@]}
        if [ "${visible_gpus}" > "${dimension_num}" ]; then
            count=${dimension_num}
        else
            count=${visible_gpus}
            sliced=("${dimensions[@]:count-1:dimension_num-visible_gpus}")
            dimensions="${sliced[*]}"
        fi

        for ((i=0; i<count; i++)); do
            export CUDA_VISIBLE_DEVICES=${gpu_ids[i]}
            python3 main.py \
                --model ${input_model} \
                --output_video_path ${output_video_path} \
		        --dataset_location ${dataset_location} \
                --ratio ${ratio} \
                --limit ${limit} \
                --dimension_list ${dimensions[i]}
                ${extra_cmd} &
            program_pid+=($!)
	        echo "Start (PID: ${program_pid[-1]}, GPU: ${i})"
        done
	    wait "${program_pid[@]}"
    else
        python3 main.py \
            --model ${input_model} \
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
        --ratio ${ratio}
}

main "$@"

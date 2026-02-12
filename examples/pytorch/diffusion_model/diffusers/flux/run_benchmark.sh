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
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --quantized_model=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      --limit=*)
          limit=$(echo $var |cut -f2 -d=)
      ;;
      --output_image_path=*)
          output_image_path=$(echo $var |cut -f2 -d=)
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
    dataset_location=${dataset_location:="captions_source.tsv"}
    limit=${limit:=-1}
    output_image_path=${output_image_path:="./tmp_imgs"}

    if [ "${topology}" = "flux_fp8" ]; then
        extra_cmd="--scheme FP8 --inference"
    elif [ "${topology}" = "flux_mxfp8" ]; then
        extra_cmd="--scheme MXFP8 --inference"
    fi

    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        gpu_list="${CUDA_VISIBLE_DEVICES:-}"
	    IFS=',' read -ra gpu_ids <<< "$gpu_list"
	    visible_gpus=${#gpu_ids[@]}
		echo "visible_gpus: ${visible_gpus}"

		python dataset_split.py --split_num ${visible_gpus} --input_file ${dataset_location} --limit ${limit}

        for ((i=0; i<visible_gpus; i++)); do
            export CUDA_VISIBLE_DEVICES=${gpu_ids[i]}

            python3 main.py \
                --model ${input_model} \
                --quantized_model_path ${tuned_checkpoint} \
                --output_image_path ${output_image_path} \
		        --eval_dataset "subset_$i.tsv" \
                ${extra_cmd} &
            program_pid+=($!)
	        echo "Start (PID: ${program_pid[-1]}, GPU: ${i})"
        done
	    wait "${program_pid[@]}"
    else
        python3 main.py \
            --model ${input_model} \
            --quantized_model_path ${tuned_checkpoint} \
            --output_image_path ${output_image_path} \
		    --eval_dataset ${dataset_location} \
			--limit ${limit} \
            ${extra_cmd}
    fi

	echo "Start calculating final score..."

    python3 main.py --output_image_path ${output_image_path} --accuracy --eval_dataset ${dataset_location}
}

main "$@"

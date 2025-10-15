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
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
       --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
       ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    tuned_checkpoint=${tuned_checkpoint:="saved_results"}

    if [ "${topology}" = "flux_fp8" ]; then
        extra_cmd="--scheme FP8 --iters 0 --dataset captions_source.tsv"
    elif [ "${topology}" = "flux_mxfp8" ]; then
        extra_cmd="--scheme MXFP8 --iters 10 --dataset captions_source.tsv"
    fi

    python3 main.py \
    	--model ${input_model} \
		--output_dir ${tuned_checkpoint} \
		--quantize \
		--inference \
		${extra_cmd}
}

main "$@"

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
      --iters=*)
          iters=$(echo $var |cut -f2 -d=)
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
    extra_cmd=""
    tuned_checkpoint=${tuned_checkpoint:="saved_results"}
    iters=${iters:=0}

    if [ "${topology}" = "llama4_mxfp4" ]; then
        extra_cmd="--fp_layers lm-head,self_attn,router,vision_model,multi_modal_projector,shared_expert --scheme MXFP4"
    fi

    python3 main.py \
    	--model ${input_model} \
	--iters ${iters} \
	--output_dir ${tuned_checkpoint} \
	${extra_cmd}
}

main "$@"

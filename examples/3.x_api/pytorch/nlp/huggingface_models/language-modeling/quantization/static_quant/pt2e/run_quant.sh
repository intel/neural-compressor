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

    if [ "${topology}" = "opt_125m_pt2e_static" ]; then
        model_name_or_path="facebook/opt-125m"
        output_dir="saved_results"
    fi
    python run_clm_no_trainer.py --model ${model_name_or_path} --quantize --output_dir ${output_dir} --tasks "lambada_openai"
}

main "$@"

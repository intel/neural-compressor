#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}

# init params
function init_params {
  topology="pokemon_diffusers"
  tuned_checkpoint="saved_results"
  model_name_or_path="lambdalabs/sd-pokemon-diffusers"
  extra_cmd=""
  batch_size=8
  approach="static"
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

    if [ "${topology}" = "pokemon_diffusers" ]; then
        model_name_or_path="lambdalabs/sd-pokemon-diffusers"
        approach="static"
    fi

    python -u ../../run_diffusion.py \
        --model_name_or_path ${model_name_or_path} \
        --tune \
        --quantization_approach ${approach} \
        --output_dir ${tuned_checkpoint} \
        --base_images ../../base_images \
        --input_text "a drawing of a gray and black dragon" \
        --calib_text "a drawing of a green pokemon with red eyes"
}

main "$@"

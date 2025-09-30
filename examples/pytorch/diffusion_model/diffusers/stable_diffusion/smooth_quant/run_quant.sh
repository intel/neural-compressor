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
      --alpha=*)
          alpha=$(echo $var |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo $var | cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=""
    model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    n_steps=20
    calib_size=10
    batch_size=1
    latent="latents.pt"
    alpha=0.44

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8 --load"
    else
        extra_cmd=$extra_cmd" --quantize"
    fi
    echo $extra_cmd

    python -u sdxl_smooth_quant.py \
        --model_name_or_path ${model_name_or_path} \
        --n_steps ${n_steps} \
        --alpha ${alpha} \
        --latent ${latent} \
        ${extra_cmd}

}

main "$@"

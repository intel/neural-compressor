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
      --model_name_or_path=*)
          model_name_or_path=$(echo $var |cut -f2 -d=)
      ;;
      --alpha=*)
          alpha=$(echo $var |cut -f2 -d=)
      ;;
      --latent=*)
          latent=$(echo $var |cut -f2 -d=)
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
    model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    n_steps=20
    calib_size=10
    batch_size=1
    latent="latents.pt"
    alpha=0.44

    python -u sdxl_smooth_quant.py \
        --model_name_or_path ${model_name_or_path} \
        --n_steps ${n_steps} \
        --quantize \
        --alpha ${alpha} \
        --latent ${latent} \
        ${extra_cmd}
}

main "$@"

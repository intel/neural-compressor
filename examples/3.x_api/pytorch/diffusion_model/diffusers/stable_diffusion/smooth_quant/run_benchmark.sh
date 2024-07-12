#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
  latent="latents.pt"
  tuned_checkpoint="./saved_results/"
  iters=200
  for var in "$@"
  do
    case $var in
      --model_name_or_path=*)
          model_name_or_path=$(echo $var | cut -f2 -d=)
      ;;
      --latent=*)
          latent=$(echo $var | cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo $var | cut -f2 -d=)
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
    extra_cmd=""
    model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    precision="fp32"
    latent="latents.pt"
    base-output-dir="./output/"
    iters=200

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    git clone https://github.com/ahmadki/mlperf_sd_inference.git
    cd mlperf_sd_inference
    mv ../main.py ./
    mv ../saved_results/ ./

    python -u main.py \
        --model-id ${model_name_or_path} \
        --quantized-unet ${tuned_checkpoint} \
        --precision ${precision} \
        --latent-path ${latent} \
        --base-output-dir ${base-output-dir} \
        --iters ${iters} \
        ${extra_cmd}
    
    mv ./output/stabilityai--stable-diffusion-xl-base-1.0__euler__20__8.0__fp32/* ./output/
    rm -rf ./output/stabilityai--stable-diffusion-xl-base-1.0__euler__20__8.0__fp32/

    python clip/clip_score.py \
        --tsv-file captions_5k.tsv \
        --image-folder ${base-output-dir} \
        --device "cpu"
}

main "$@"

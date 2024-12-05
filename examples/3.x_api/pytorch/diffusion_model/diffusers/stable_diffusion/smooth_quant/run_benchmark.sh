ert!/bin/bash
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
      --iters=*)
          iters=$(echo $var | cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo $var | cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var | cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd="--load"
    model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"
    precision="fp32"
    latent="latents.pt"
    base_output_dir="./output/"

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd

    if [[ ${mode} == "performance" ]]; then
      extra_cmd=$extra_cmd" --performance"
      echo $extra_cmd

      python -u sdxl_smooth_quant.py \
        --model_name_or_path ${model_name_or_path} \
        --latent ${latent} \
        ${extra_cmd}
    else
      echo $extra_cmd

      python -u sdxl_smooth_quant.py \
        --model_name_or_path ${model_name_or_path} \
        --latent ${latent} \
        ${extra_cmd}
        
      REPO_URL="https://github.com/ahmadki/mlperf_sd_inference.git"
      TARGET_DIR="mlperf_sd_inference"

      if [ -d "$TARGET_DIR" ]; then
        echo "Directory $TARGET_DIR already exists. Skipping git clone."
      else
        git clone "$REPO_URL" "$TARGET_DIR"
      fi

      cd mlperf_sd_inference
      cp ../main.py ./
      if [ -d "../saved_results/" ]; then
        cp -r ../saved_results/ ./
      fi
      
      python -u main.py \
        --model-id ${model_name_or_path} \
        --quantized-unet ${tuned_checkpoint} \
        --precision ${precision} \
        --latent-path ${latent} \
        --base-output-dir ${base_output_dir} \
        --iters ${iters} \
        ${extra_cmd}

      mv ./output/stabilityai--stable-diffusion-xl-base-1.0__euler__20__8.0__fp32/* ./output/
      rm -rf ./output/stabilityai--stable-diffusion-xl-base-1.0__euler__20__8.0__fp32/

      python clip/clip_score.py \
          --tsv-file captions_5k.tsv \
          --image-folder ${base_output_dir} \
          --device "cpu"
      
      cd ..
    fi

}

main "$@"

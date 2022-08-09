#!/bin/bash
set -x

function main {

  init_params "$@"
  extra_cmd=" "
  set_args
  define_mode
  run_benchmark

}

# init params
function init_params {
  iters=100
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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

function define_mode {

    if [[ ${mode} == "accuracy" ]]; then
      echo "For TF OOB models, there is only benchmark mode!, num iter is: ${iters}"
      exit 1
    elif [[ ${mode} == "benchmark" ]]; then
      mode_cmd=" --benchmark "
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

function set_args {

  models_need_name=(
  --------
  AttRec
  CRNN
  CapsuleNet
  CenterNet
  CharCNN
  COVID-Net
  DLRM
  Time_series_LSTM
  Hierarchical_LSTM
  MANN
  MiniGo
  TextCNN
  TextRNN
  aipg-vdcnn
  arttrack-coco-multi
  arttrack-mpii-single
  context_rcnn_resnet101_snapshot_serenget
  deepspeech
  deepvariant_wgs
  dense_vnet_abdominal_ct
  east_resnet_v1_50
  efficientnet-b0
  efficientnet-b0_auto_aug
  efficientnet-b5
  efficientnet-b7_auto_aug
  facenet-20180408-102900
  handwritten-score-recognition-0003
  license-plate-recognition-barrier-0007
  optical_character_recognition-text_recognition-tf
  pose-ae-multiperson
  pose-ae-refinement
  resnet_v2_200
  show_and_tell
  text-recognition-0012
  vggvox
  wide_deep
  yolo-v3-tiny
  NeuMF
  PRNet
  DIEN_Deep-Interest-Evolution-Network
  --------
  )

  models_need_disable_optimize=(
  --------
  COVID-Net
  Time_series_LSTM
  CRNN
  efficientnet-b0
  efficientnet-b0_auto_aug
  efficientnet-b5
  efficientnet-b7_auto_aug
  vggvox
  ava-person-vehicle-detection-stage2-2_0_0
  DIEN_Deep-Interest-Evolution-Network
  --------
  )

  # neural_compressor graph_def
  models_need_nc_graphdef=(
  --------
  pose-ae-multiperson
  pose-ae-refinement
  centernet_hg104
  DETR
  Elmo
  Time_series_LSTM
  Unet
  WD
  ResNest101
  ResNest50
  ResNest50-3D
  adversarial_text
  Attention_OCR
  GPT2
  Parallel_WaveNet
  PNASNet-5
  VAE-CF
  DLRM
  Deep_Speech_2
  --------
  )

  # neural_compressor need output for ckpt
  if [ "${topology}" == "adversarial_text" ];then
    extra_cmd+=" --output_name Identity "
  elif [ "${topology}" == "Attention_OCR" ];then
    extra_cmd+=" --output_name AttentionOcr_v1/predicted_text "
  elif [ "${topology}" == "AttRec" ];then
    extra_cmd+=" --output_name  "
  elif [ "${topology}" == "GPT2" ];then
    extra_cmd+=" --output_name strided_slice "
  elif [ "${topology}" == "Parallel_WaveNet" ];then
    extra_cmd+=" --output_name truediv_1 "
  elif [ "${topology}" == "PNASNet-5" ];then
    extra_cmd+=" --output_name final_layer/FC/BiasAdd "
  elif [ "${topology}" == "VAE-CF" ];then
    extra_cmd+=" --output_name private_vae_graph/sequential_1/decoder_20024/BiasAdd "
  fi

}

# run_tuning
function run_benchmark {
    extra_cmd+=" --num_iter ${iters} --num_warmup 10 "

    if [[ "${models_need_name[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need model name!"
      extra_cmd+=" --model_name ${topology} "
    fi
    if [[ "${models_need_disable_optimize[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need to disable optimize_for_inference!"
      extra_cmd+=" --disable_optimize "
    fi
    if [[ "${models_need_nc_graphdef[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need neural_compressor graph_def!"
      extra_cmd+=" --use_nc "
    fi
    if [[ "${models_need_use_nc_optimizer[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need to use pre optimizer!"
      extra_cmd+=" --use_nc_optimize "
    fi

    python tf_benchmark.py \
            --model_path ${input_model} \
            ${extra_cmd} \
            ${mode_cmd}
}

main "$@"

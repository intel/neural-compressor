#!/bin/bash
set -x

function main {

  init_params "$@"
  extra_cmd=" "
  set_args
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
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# set exe args
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
  EfficientDet-D2-768x768
  EfficientDet-D4-1024x1024
  centernet_hg104
  --------
  )

  models_need_disable_optimize=(
  --------
  CRNN
  COVID-Net
  Time_series_LSTM
  efficientnet-b0
  efficientnet-b0_auto_aug
  efficientnet-b5
  efficientnet-b7_auto_aug
  vggvox
  --------
  )

  # bs !=1 when tuning
  models_need_bs16=(
  --------
  icnet-camvid-ava-0001
  icnet-camvid-ava-sparse-30-0001
  icnet-camvid-ava-sparse-60-0001
  --------
  )
  models_need_bs32=(
  --------
  adv_inception_v3
  ens3_adv_inception_v3
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
  AttRec
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
function run_tuning {
    input="input"
    output="predict"
    extra_cmd+=' --num_warmup 10 -n 500 '

    if [[ "${models_need_name[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need model name!"
      extra_cmd+=" --model_name ${topology} "
    fi
    if [[ "${models_need_disable_optimize[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need to disable optimize_for_inference!"
      extra_cmd+=" --disable_optimize "
    fi
    if [[ "${models_need_bs16[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need to set bs = 16!"
      extra_cmd+=" -b 16 "
    fi
    if [[ "${models_need_bs32[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need to set bs = 32!"
      extra_cmd+=" -b 32 "
    fi
    if [[ "${models_need_nc_graphdef[@]}"  =~ " ${topology} " ]]; then
      echo "$topology need neural_compressor graph_def!"
      extra_cmd+=" --use_nc "
    fi

    python tf_benchmark.py \
            --model_path ${input_model} \
            --output_path ${output_model} \
            --tune \
            ${extra_cmd}

}

main "$@"


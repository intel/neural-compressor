#!/usr/bin/env bash
### This file is originally from: [mlcommons repo](https://github.com/mlcommons/inference/tree/r0.5/others/cloud/single_stage_detector/download_model.sh)
CHECKPOINT_DIR=${CHECKPOINT_DIR-$PWD}

dir=$(pwd)
mkdir -p ${CHECKPOINT_DIR}/pretrained; cd ${CHECKPOINT_DIR}/pretrained
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z'  -O 'resnet34-ssd1200.pth'
cd $dir
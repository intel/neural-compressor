#!/bin/bash

BUILD_DIR=build
DATA_DIR=${BUILD_DIR}/data
MODEL_DIR=${DATA_DIR}/bert_tf_v1_1_large_fp32_384_v2

if [ ! -d ${MODEL_DIR} ]; then
    mkdir ${MODEL_DIR};
fi

if [ ! -f ${MODEL_DIR}/model.ckpt-5474.data-00000-of-00001 ]; then
    wget -O ${MODEL_DIR}/model.ckpt-5474.data-00000-of-00001 https://zenodo.org/record/3733868/files/model.ckpt-5474.data-00000-of-00001?download=1
fi
if [ ! -f ${MODEL_DIR}/model.ckpt-5474.index ]; then
    wget -O ${MODEL_DIR}/model.ckpt-5474.index https://zenodo.org/record/3733868/files/model.ckpt-5474.index?download=1
fi
if [ ! -f ${MODEL_DIR}/model.ckpt-5474.meta ]; then
    wget -O ${MODEL_DIR}/model.ckpt-5474.meta https://zenodo.org/record/3733868/files/model.ckpt-5474.meta?download=1
fi

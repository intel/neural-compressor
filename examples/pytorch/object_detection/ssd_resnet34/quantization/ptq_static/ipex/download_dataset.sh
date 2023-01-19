#!/usr/bin/env bash
### This file is originally from: [mlcommons repo](https://github.com/mlcommons/inference/tree/r0.5/others/cloud/single_stage_detector/download_dataset.sh)
DATASET_DIR=${DATASET_DIR-$PWD}

dir=$(pwd)
mkdir -p ${DATASET_DIR}/coco; cd ${DATASET_DIR}/coco
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd $dir
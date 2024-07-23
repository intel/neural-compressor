#!/bin/bash
# set -x

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
unzip -o coco2017labels.zip
rm coco2017labels.zip

cd coco
mkdir images
cd images
wget http://images.cocodataset.org/zips/val2017.zip
unzip -o val2017.zip
rm val2017.zip

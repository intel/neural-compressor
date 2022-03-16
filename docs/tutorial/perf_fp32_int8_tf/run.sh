#!/bin/bash


#use PyPi to setup a virtual environment
ENV_NAME=env_intel_tf
if [ ! -d $ENV_NAME ]; then
    echo "Create env $ENV_NAME ..."
    bash set_env.sh
else
    echo "Already created env $ENV_NAME, skip craete env"
fi

source $ENV_NAME/bin/activate

INT8_FILE="resnet50_int8_pretrained_model.pb"
FP32_FILE="resnet50_fp32_pretrained_model.pb"

#download both models, don't download if already exist
if [ ! -f $INT8_FILE ]; then
    echo "Downloading int8 model..."
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_int8_pretrained_model.pb
else
    echo "Already downloaded int8 model, skip downloading"
fi

if [ ! -f $FP32_FILE ]; then
    echo "Downloading fp32 model..."
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_fp32_pretrained_model.pb
else
    echo "Already downloaded fp32 model, skip downloading"
fi

#run inference and make images from them
echo "Test Performance of FP32 and INT8 Models"
python test_performance.py $FP32_FILE $INT8_FILE

echo "Compare the Performance of FP32 and INT8 Models"
python compare_result.py


if [[ $? -eq 0 ]]
then
  echo "The sample is finished. Please check the PNG files to see the performance!"
else
  echo "The sample is not finished correctly"
fi


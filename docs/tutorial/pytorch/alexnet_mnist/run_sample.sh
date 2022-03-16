#!/bin/bash

echo "Note, please enable running environment before running this script."


echo "Train Model by Pytorch with Fashion MNIST"
python train_alexnet_fashion_mnist.py

FP32_FILE="alexnet_mnist_fp32_mod.pth"
if [ ! -f $FP32_FILE ]; then
    echo "$FP32_FILE not exists."
    echo "Train AlexNet model is fault, exit!"
    exit 1
else
    echo "Training is finished"
fi
 
echo "Quantize Model by Intel Neural Compressor"
python inc_quantize_model.py

INT8_FOLDER="alexnet_mnist_int8_mod"
if [ ! -d $INT8_FOLDER ]; then
    echo "$INT8_FOLDER not exists."
    echo "Quantize FP32 model is fault, exit!"
    exit 1
else
    echo "Quantization is finished"
fi

echo "Execute the profiling_inc.py with FP32 model file"
python profiling_inc.py --input-graph=./alexnet_mnist_fp32_mod.pth --index=32
echo "FP32 performance test is finished"

echo "Execute the profiling_inc.py with INT8 model file"
python profiling_inc.py --input-graph=./alexnet_mnist_int8_mod --index=8
echo "INT8 performance test is finished"

echo "Compare the Performance of FP32 and INT8 Models"
python compare_perf.py
echo "Please check the PNG files to see the performance!"

if [[ $? -eq 0 ]]; then
  echo "This demo is finished successfully!"
else
  echo "This demo is fault!"
fi

echo "Thank you!"
exit 0

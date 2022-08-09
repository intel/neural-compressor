#!/bin/bash

echo "Note, please enable running environment before running this script."

echo "Enable Intel Optimized TensorFlow 2.6.0 and newer by setting environment variable TF_ENABLE_ONEDNN_OPTS=1"
echo "That will accelerate training and inference, and  it's mandatory requirement of running Intel® Neural Compressor quantize Fp32 model or deploying the quantized model."

export TF_ENABLE_ONEDNN_OPTS=1

echo "Train Model by Keras/Tensorflow with MNIST"
python keras_tf_train_mnist.py

FP32_FILE="fp32_frozen.pb"
if [ ! -f $FP32_FILE ]; then
    echo "$FP32_FILE not exists."
    echo "Train AlexNet model is fault, exit!"
    exit 1
else
    echo "Training is finished"
fi
 
echo "Quantize Model by Intel Neural Compressor"
python inc_quantize_model.py

INT8_FILE="alexnet_int8_model.pb"
if [ ! -f $INT8_FILE ]; then
    echo "$INT8_FILE not exists."
    echo "Quantize FP32 model is fault, exit!"
    exit 1
else
    echo "Quantization is finished"
fi

echo "Execute the profiling_inc.py with FP32 model file"
python profiling_inc.py --input-graph=./fp32_frozen.pb --omp-num-threads=4 --num-inter-threads=1 --num-intra-threads=4 --index=32
echo "FP32 performance test is finished"

echo "Execute the profiling_inc.py with INT8 model file"
python profiling_inc.py --input-graph=./alexnet_int8_model.pb --omp-num-threads=4 --num-inter-threads=1 --num-intra-threads=4 --index=8
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

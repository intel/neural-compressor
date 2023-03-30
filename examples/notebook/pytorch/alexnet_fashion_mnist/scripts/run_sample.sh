#!/bin/bash

echo "Note: This script relies on relevant Python environment for its execution. Refer README.md for more details."

echo "Train Pytorch AlexNet model on Fashion MNIST dataset"
python python_src/train_alexnet_fashion_mnist.py

FP32_FILE="../output/alexnet_mnist_fp32_mod.pth"
if [ ! -f $FP32_FILE ]; then
    echo "$FP32_FILE - model file does not exist"
    echo "Model training failed, exiting!. Check error logs for details"
    exit 1
else
    echo "Model training has completed successfully"
fi
 
echo "Quantize Model using Intel Neural Compressor"
python python_src/inc_quantize_model.py

INT8_FOLDER="../output/alexnet_mnist_int8_mod"
if [ ! -d $INT8_FOLDER ]; then
    echo "$INT8_FOLDER not exists."
    echo "Model quantization has failed, exiting!. Check error logs for details"
    exit 1
else
    echo "Model quantization has completed successfully"
fi

echo "Execute the profiling_inc.py with FP32 model file"
python python_src/profiling_inc.py --input-graph=../output/alexnet_mnist_fp32_mod.pth --index=32
echo "FP32 model performance test has completed successfully"

echo "Execute the profiling_inc.py with INT8 model file"
python python_src/profiling_inc.py --input-graph=../output/alexnet_mnist_int8_mod --index=8
echo "INT8 model performance test has completed successfully"

echo "Comparing  the Performance of FP32 and INT8 Models"
python python_src/compare_perf.py
echo "Check the output PNG files for performance comparison!"

if [[ $? -eq 0 ]]; then
  echo "Demo execution completed successfully! Check output directory for results."
else
  echo "Demo execution has failed! Check error logs for more details."
fi

echo "Thank you!"
exit 0

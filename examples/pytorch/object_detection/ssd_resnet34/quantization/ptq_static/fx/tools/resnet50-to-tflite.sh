#!/bin/bash

# convert resnet50 to tflite


model=resnet50_v1
tfmodel="$model.pb"
url=https://zenodo.org/record/2535873/files/$tfmodel

if [ ! -r $local ]; then
    wget -o $local -q  $url
fi

tflite_convert --graph_def_file $tfmodel --output_file $model.tflite \
    --input_arrays input_tensor \
    --output_arrays ArgMax,softmax_tensor 

tflite_convert --graph_def_file  $tfmodel --output_file $model"_quant.tflite" \
    --input_arrays input_tensor \
    --output_arrays ArgMax  \
    --inference_type QUANTIZED_UINT8 --inference_input_type QUANTIZED_UINT8 \
    --input_shape=1,224,224,3 \
    --mean_values=128 \
    --std_dev_values=128 \
    --default_ranges_min=0 \
    --default_ranges_max=6

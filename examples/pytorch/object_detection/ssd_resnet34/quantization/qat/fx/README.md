Step-by-Step
============

This document list steps of reproducing Intel Optimized PyTorch ssd_resnet34 300*300 models tuning results via Neural Compressor.

Our example comes from MLPerf Training Inference Suite


# Prerequisite

### 1. Installation

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

  ```shell
  cd examples/pytorch/object_detection/ssd_resnet34/quantization/qat/fx
  pip install -r requirements.txt
  pip install "git+https://github.com/mlperf/logging.git"
  ```

### 2. Download Dataset

  ```shell
  sh download_dataset.sh
  ```

### 3. Train the Model
Follow the instructions on https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd to train the model.

# Run

### 1. Enable ssd_resnet34 example with quant aware training strategy of Neural Compressor.

  The changes made are as follows:
  1. add conf.yaml:
    This file contains the configuration of quantization.
  2. add ssd/main.py:\
    we add the eval_func and training_func_for_nc with reference to https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/train.py \
    we import neural_compressor in it.
  3. edit ssd/ssd300.py:
    we replace view() with reshape() in function bbox_view().

### 2. To get the tuned model and its accuracy: 

    bash run_tuning.sh  --topology=resnet34 --dataset_location=coco/ --input_model=$trained model path$  --output_model=saved_results

### 3. To get the benchmark of tuned model, includes Batch_size and Throughput: 

    bash run_benchmark.sh --topology=resnet34 --dataset_location=coco/ --input_model=$trained model path$ --mode=benchmark --int8=true/false



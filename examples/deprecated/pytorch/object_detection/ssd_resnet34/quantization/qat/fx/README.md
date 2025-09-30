Step-by-Step
============

This document lists steps of reproducing Intel Optimized PyTorch ssd_resnet34 300*300 models tuning results via Neural Compressor.

Our example comes from MLPerf Training Inference Suite


# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend. We recommend to use Python 3.10.

  ```shell
  cd examples/pytorch/object_detection/ssd_resnet34/quantization/qat/fx
  pip install -r requirements.txt
  pip install "git+https://github.com/mlperf/logging.git"
  ```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Download Dataset

  ```shell
  sh download_dataset.sh
  ```

## 3. Train the Model

Follow the instructions on https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd to train the model.

# Quantization

## 1. Enable ssd_resnet34 example with quant aware training strategy of Neural Compressor.

  The changes made are as follows:
  1. add ssd/main.py:\
    We import neural_compressor and pass `QuantizationAwareTrainingConfig` to the quant aware training process.
    We then add the eval_func and training_func_for_nc with reference to https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/train.py \
    we import neural_compressor in it.
  2. edit ssd/ssd300.py:
    we replace view() with reshape() in function bbox_view().

## 2. To get the tuned model and its accuracy:

    bash run_quant.sh  --topology=resnet34 --dataset_location=coco/ --input_model=$trained model path$  --output_model=saved_results

## 3. To get the benchmark of tuned model, includes Batch_size and Throughput:

    bash run_benchmark.sh --topology=resnet34 --dataset_location=coco/ --input_model=$trained model path$ --mode=performance --int8=true/false



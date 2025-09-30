Step-by-Step
============

This document lists steps of reproducing Intel Optimized PyTorch ssd_resnet34 models tuning results via Neural Compressor.

Our example comes from MLPerf Inference Benchmark Suite


# Prerequisite

## 1. Environment

PyTorch 1.8 or higher version is needed with pytorch_fx backend. We recommend to use Python 3.8.

```shell
cd examples/pytorch/object_detection/ssd_resnet34/quantization/ptq/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

Check your gcc version with command : **gcc -v**

GCC5 or above is needed.

  ```shell
  bash prepare_loadgen.sh
  ```

## 2. Prepare Dataset

- Step1: Download COCO2017 dataset and extract it.
- Step2: Upscale COCO2017 dataset image size into 1200x1200 with prepare_dataset.sh

| dataset | download link | 
| ---- | ---- | 
| coco (validation) | http://images.cocodataset.org/zips/val2017.zip | 
| coco (annotations) | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |

  ```shell
  cd examples/pytorch/object_detection/ssd_resnet34/quantization/ptq/fx
  bash prepare_dataset.sh --origin_dir=origin_dataset --convert_dir=convert_dataset
  ```
  Make sure origin_dataset (COCO2017) have two folder: val2017 and annotations.


## 3. Prepare pre-trained model

  ```shell
  cd examples/pytorch/object_detection/ssd_resnet34/quantization/ptq/fx
  wget https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch
  ```

# Quantization

## 1. Enable ssd_resnet34 example with the auto dynamic quantization strategy of Neural Compressor.

  The changes made are as follows:
  1. edit python/main.py:
    we import neural_compressor in it and pass `PostTrainingQuantConfig` to the quantization process.
  2. edit python/model/ssd_r34.py:
    we wrap functions with @torch.fx.wrap to avoid ops cannot be traced by fx mode.

## 2. To get the tuned model and its accuracy:

    bash run_quant.sh  --topology=ssd-resnet34 --dataset_location=./convert_dataset --input_model=./resnet34-ssd1200.pytorch  --output_model=./saved_results

## 3. To get the benchmark of tuned model, includes Batch_size and Throughput:

    bash run_benchmark.sh --topology=ssd-resnet34 --dataset_location=./convert_dataset --input_model=./resnet34-ssd1200.pytorch --config=./saved_results --mode=performance --int8=true/false

## 4. The following is the brief output information:

Left part is accuracy/percentage, right part is time_usage/second.

sampling_size: 50
FP32 baseline is: [19.6298, 3103.3418]
Pass quantize model elapsed time: 76469.05 ms
Tune 1 result is: [19.1733, 763.7865]
Pass quantize model elapsed time: 22288.36 ms
Tune 2 result is: [19.4817, 861.9649]

|       | Batch size | Latency | Throughput |
| ----- | ---------- | ------- | ----------- |
| fp32  | 1 | 878.225 ms | 1.139 samples/sec |
| int8  | 1 |  97.111 ms | 10.298 samples/sec |

sampling_size: 500
FP32 baseline is: [19.6298, 3103.3418]  
Pass quantize model elapsed time: 480769.63 ms  
Tune 1 result is: [19.0617, 649.5251]  
Pass quantize model elapsed time: 215259.43 ms  
Tune 2 result is: [19.5257, 636.5329]  
···  

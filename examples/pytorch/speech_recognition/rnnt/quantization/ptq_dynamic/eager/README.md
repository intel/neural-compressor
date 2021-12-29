Step-by-Step
============

This document list steps of reproducing Intel Optimized PyTorch RNNT models tuning results via Neural Compressor.

Our example comes from MLPerf Inference Benchmark Suite


# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```shell
  cd examples/pytorch/speech_recognition/rnnt/quantization/ptq_dynamic/eager
  pip install -r requirements.txt
  ```
  Check your gcc version with command : **gcc -v**

  GCC5 or above is needed.

  ```shell
  bash prepare_loadgen.sh
  ```

### 2. Prepare Dataset

  ```shell
  cd examples/pytorch/speech_recognition/rnnt/quantization/ptq_dynamic/eager
  bash prepare_dataset.sh --download_dir=origin_dataset --convert_dir=convert_dataset
  ```

  Prepare_dataset.sh contains two stages:
  - stage1: download LibriSpeech/dev-clean dataset and extract it.
  - stage2: convert .flac file to .wav file

### 3. Prepare pre-trained model

  ```shell
  cd examples/pytorch/speech_recognition/rnnt/quantization/ptq_dynamic/eager
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O rnnt.pt
  ```

# Run

### 1. Enable RNNT example with the auto dynamic quantization strategy of Neural Compressor.

  The changes made are as follows:
  1. add conf.yaml:
    This file contains the configuration of quantization.
  2. run.py->run_tune.py:
    we added neural_compressor support in it.
  3. edit pytorch_SUT.py:
    remove jit script convertion
  4. edit pytorch/decoders.py:
    remove assertion of torch.jit.ScriptModule

### 2. To get the tuned model and its accuracy: 

    bash run_tuning.sh --dataset_location=convert_dataset --input_model=./rnnt.pt --output_model=saved_results

### 3. To get the benchmark of tuned model, includes Batch_size and Throughput: 

    bash run_benchmark.sh --dataset_location=convert_dataset --input_model=./rnnt.pt --mode=benchmark/accuracy --int8=true/false

### 4. The following is the brief output information:

Left part is accuracy/percentage, right part is time_usage/second.

  - FP32 baseline is: [92.5477, 796.7552]. 
  - Tune 1 result is: [91.5872, 1202.2529]
  - Tune 2 result is: [91.5894, 1201.3231]
  - Tune 3 result is: [91.5195, 1211.5965]
  - Tune 4 result is: [91.6030, 1218.2211]
  - Tune 5 result is: [91.4812, 1169.5080]
  - ...


Step-by-Step
============

This document is used to list steps of reproducing Intel Optimized TensorFlow OOB models tuning zoo result.

> **Note**: 
> Most of those models are both supported in Intel optimized TF 1.15.x and Intel optimized TF 2.x. We use 1.15.2 as an example.

# Prerequisite

### 1. Installation
  Recommend python 3.6 or higher version.

  ```Shell
  pip install -r requirements.txt
  
  ```

### 2. Prepare Dataset

  We use dummy data to do benchmarking with Tensorflow OOB models.

### 3. Prepare pre-trained model


# Run
### run tuning

```bash
./run_tune.sh --topology=${model_topology} --dataset_location= --input_model=${model_path} --output_model=${output_model_path}
```

### run benchmarking

```bash
./run_benchmarking.sh --topology=${model_topology} --dataset_location= --input_model=${model_path} --mode=benchmark --batch_size=1 --iters=200
```
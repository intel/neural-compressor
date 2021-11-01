Step-by-Step
============

# Prerequisite

### 1. Installation
  ```shell
  conda create -n <env name> python=3.7
  conda activate <env name>
  cd <nc_folder>/examples/deepengine/nlp/bert_large
  pip install 1.15.0 up2 from links below:
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset and Model
### 2.1 Prepare Dataset
  ```shell
  bash prepare_dataset.sh
  ```

### 2.2 Download TensorFlow model (The model will be in build/data/bert_tf_v1_1_large_fp32_384_v2 folder):
  ```shell
  bash prepare_model.sh
  ```

### 2.3 Get the frozen pb model (The model.pb will be in build/data):
  ```shell
  python tf_freeze_bert.py
  ```

### Run

### 1. To get the tuned model and its accuracy:
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --tune
  ```
  or run shell
  ```shell
  bash run_tuning.sh --config=bert.yaml --input_model=build/data/model.pb --output_model=ir --dataset_location=build/data
  ```

### 2. To get the benchmark of tuned model:
  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=accuracy --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=build/data --batch_size=1 --mode=accuracy
  ```

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=performance --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=build/data --batch_size=1 --mode=performance
  ```
  
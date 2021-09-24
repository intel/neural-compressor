Step-by-Step
============

# Prerequisite

### 1. Installation
  ```shell
  conda create -n <env name> python=3.7
  conda activate <env name>
  cd <nc_folder>/examples/deepengine/nlp/bert_base_mrpc
  pip install 1.15.0 up2 from links below:
  https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
  pip install -r requirements.txt
  ```

### 2. Prepare Dataset and pretrained model
### 2.1 Get dataset
  ```shell
  python prepare_dataset.py --tasks='MRPC' --output_dir=./data
  ```

### 2.2 Get model
  ```shell
  bash prepare_model.sh --dataset_location=./data --output_dir=./model
  ```

### Run

### 1. To get the tuned model and its accuracy:
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --tune
  ```
  or run shell
  ```shell
  bash run_tuning.sh --config=bert.yaml --input_model=model/bert_base_mrpc.pb --output_model=ir --dataset_location=data
  ```

### 2. To get the benchmark of tuned model:
  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=accuracy --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=1 --mode=accuracy
  ```

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=performance --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=1 --mode=performance
  ```
  
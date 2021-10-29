# Step-by-Step

# Prerequisite

### 1. Installation

```shell
conda create -n <env name> python=3.7
conda activate <env name>
cd <neural_compressor_folder>/examples/engine/nlp/bert_base_sparse_mrpc
pip install -r requirements.txt
```

### 2. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --tasks='MRPC' --output_dir=./data
```

### 2.2 Get model

```shell
bash prepare_model.sh 
```

### Run

### 1. To get the tuned model and its accuracy:
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --tune
  ```
  or run shell
  ```shell
  bash run_tuning.sh --config=bert.yaml --input_model=bert_base_sparse_mrpc.onnx --output_model=ir --dataset_location=data
  ```

### 2. To get the benchmark of tuned model:
  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=accuracy --batch_size=8
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=8 --mode=accuracy
  ```

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=performance --batch_size=8
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=8 --mode=performance
  ```


# Step-by-Step

# Prerequisite

### 1. Installation

```shell
conda create -n <env name> python=3.7
conda activate <env name>
cd <neural_compressor_folder>/examples/engine/nlp/paraphrase_xlm_r_multilingual_v1_stsb
pip install -r requirements.txt
```

### 2. Prepare Dataset and pretrained model

### 2.1 Get dataset

```shell
python prepare_dataset.py --output_dir=./data
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
  bash run_tuning.sh --config=bert.yaml --input_model=paraphrase_xlm_r_multilingual_v1_stsb.onnx --output_model=ir --dataset_location=data
  ```

### 2. To get the benchmark of tuned model:
  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=accuracy --batch_size=4
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=4 --mode=accuracy
  ```

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=performance --batch_size=4
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert.yaml --input_model=ir --dataset_location=data --batch_size=4 --mode=performance
  ```

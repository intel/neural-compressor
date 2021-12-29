# Step-by-Step

# Prerequisite

### 1. Installation
1.1 Install python environment
```shell
conda create -n <env name> python=3.7
conda activate <env name>
cd <neural_compressor_folder>/examples/engine/nlp/mrpc/bert_base_sparse
pip install -r requirements.txt
```
1.2 Install C++ environment (Optional)
Install engine according to [Engine](../../../../../docs/engine.md) if need the performance in C++.
Preload libiomp5.so can improve the performance when bs=1.
```
export LD_PRELOAD=<path_to_libiomp5.so>
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
  or run C++
  The warmup below is recommended to be 1/10 of iterations and no less than 3.
  ```
  export GLOG_minloglevel=2
  export OMP_NUM_THREADS=<cpu_cores>
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  export UNIFIED_BUFFER=1
  numactl -C 0-<cpu_cores-1> <neural_compressor_folder>/engine/bin/inferencer --batch_size=<batch_size> --iterations=<iterations> --w=<warmup> --seq_len=128 --config=./ir/conf.yaml --weight=./ir/model.bin
  ```

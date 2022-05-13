Step-by-Step
============

# Prerequisite

### 1. Installation
1.1 Install python environment
```shell
conda create -n <env name> python=3.7
conda activate <env name>
cd <nc_folder>/examples/baremetal/nlp/mrpc/bert_base
pip install 1.15.0 up2 from links below:
https://storage.googleapis.com/intel-optimized-tensorflow/intel_tensorflow-1.15.0up2-cp37-cp37m-manylinux2010_x86_64.whl
pip install -r requirements.txt
```
Preload libiomp5.so can improve the performance when bs=1.
```
export LD_PRELOAD=<path_to_libiomp5.so>
```
Preloading libjemalloc.so can improve the performance. It has been built in third_party/jemalloc/lib.
```
export LD_PRELOAD=<path_to_libjemalloc.so>
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
  bash run_tuning.sh --config=bert_static.yaml --input_model=model/bert_base_mrpc.pb --output_model=ir --dataset_location=data
  ```

### 2. To get the benchmark of tuned model:
  2.1 accuracy
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=accuracy --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert_static.yaml --input_model=ir --dataset_location=data --batch_size=1 --mode=accuracy
  ```

  2.2 performance
  run python
  ```shell
  GLOG_minloglevel=2 python run_engine.py --input_model=./ir --benchmark --mode=performance --batch_size=1
  ```
  or run shell
  ```shell
  bash run_benchmark.sh --config=bert_static.yaml --input_model=ir --dataset_location=data --batch_size=1 --mode=performance
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

# Step-by-Step

This example quantizes and validates the accuracy of Llama4.

# Prerequisite

## 1. Environment

```shell
docker run -d --gpus all -v ... --shm-size=100g --name llama4 -it nvcr.io/nvidia/pytorch:25.05-py3 /bin/bash
docker exec -it llama4 bash
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor/examples/pytorch/multimodal-modeling/quantization/auto_round/llama4
# Use `INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@v3.6rc` for the latest updates before neural-compressor v3.6 release
pip install neural-compressor-pt==3.6
# Use `pip install git+https://github.com/intel/auto-round.git@v0.8.0rc2` for the latest updates before auto-round v0.8.0 release
pip install auto-round==0.8.0
bash setup.sh
```

## 2. Prepare Model

```shell
hf download meta-llama/Llama-4-Scout-17B-16E-Instruct --local-dir Llama-4-Scout-17B-16E-Instruct
```

# Run

## 1. Quantization

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quant.sh --topology=llama4_mxfp4 --input_model=Llama-4-Scout-17B-16E-Instruct/
```


## 2. Benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --topology=llama4_mxfp4 --input_model=saved_results --tasks=piqa --batch_size=1 --tp_size=4
```

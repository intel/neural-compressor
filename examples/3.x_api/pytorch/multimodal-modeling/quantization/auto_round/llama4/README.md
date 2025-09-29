# Step-by-Step

This example quantizes and validates the accuracy of Llama4.

# Prerequisite

## 1. Environment

```shell
docker run -d --gpus all -v ... --shm-size=100g --name llama4 -it nvcr.io/nvidia/pytorch:25.05-py3 /bin/bash
docker exec -it llama4 bash
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor/examples/3.x_api/pytorch/multimodal-modeling/quantization/auto_round/llama4
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
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --topology=llama4_mxfp4 --input_model=saved_results/Llama-4-Scout-17B-16E-Instruct-w4g32/ --tasks=piqa --batch_size=1 --tp_size=4
```

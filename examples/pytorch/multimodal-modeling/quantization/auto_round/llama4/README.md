# Step-by-Step

This example quantizes and validates the accuracy of Llama4.

# Prerequisite

## 1. Environment

```shell
docker run -d --gpus all -v ... --shm-size=100g --name llama4 -it nvcr.io/nvidia/pytorch:25.08-py3 /bin/bash
docker exec -it llama4 bash
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor/examples/pytorch/multimodal-modeling/quantization/auto_round/llama4
# Use `INC_PT_ONLY=1 pip install git+https://github.com/intel/neural-compressor.git@master` for the latest updates before neural-compressor v3.7 release
pip install neural-compressor-pt==3.7
# Use `pip install git+https://github.com/intel/auto-round.git@main` for the latest updates before auto-round v0.9.3 release
pip install auto-round==0.9.3
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

> Note: You can also enable static quantization for KV cache by adding `--static_kv_dtype fp8` argument to `main.py`， or `--static_kv_dtype=fp8` argument to `run_quant.sh` and `run_benchmark.sh`.
>
> You can also enable static quantization for attention by adding `--static_attention_dtype fp8` argument to `main.py`， or `--static_attention_dtype=fp8` argument to `run_quant.sh` and `run_benchmark.sh`. When enabled, it automatically sets KV cache dtype to fp8 as well.

## 2. Benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh --topology=llama4_mxfp4 --input_model=saved_results --tasks=piqa --batch_size=1 --tp_size=4
```

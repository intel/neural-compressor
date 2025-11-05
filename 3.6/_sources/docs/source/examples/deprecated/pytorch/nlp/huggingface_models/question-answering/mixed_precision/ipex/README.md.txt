Step-by-Step
============
This document describes the step-by-step instructions for reproducing Huggingface models with IPEX backend MixedPrecision results with Intel® Neural Compressor.
> Note: IPEX version >= 1.10

# Prerequisite

## Environment
Recommend python 3.6 or higher version.
```shell
cd examples/pytorch/nlp/huggingface_models/question-answering/mixed_precision/ipex
pip install -r requirements.txt
```
> Note: Intel® Extension for PyTorch* has PyTorch version requirement. 

# Run 
## Mixed Precision
If IPEX version is equal or higher than 1.12, please install transformers 4.19.0.  
IPEX doesn't support accuracy-driven mixed precision, so the model convert just execute once based on the framework capability.
```shell
python run_qa.py 
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --no_cuda \
    --optimize \
    --output_dir ./saved_results
```
> NOTE: 
> /path/to/checkpoint/dir is the path to finetune output_dir

## Benchmark
```Shell
# run optimized performance
bash run_benchmark.sh --mode=performance --batch_size=1 --topology=distilbert_base_ipex --optimized=true --iters=500
# run optimized accuracy
bash run_benchmark.sh --mode=accuracy --batch_size=8 --topology=distilbert_base_ipex --optimized=true
```

# Step by step
This document describes the step-by-step instructions for reproducing bert-large and distilbert-base models with IPEX backend tuning results with Intel® Neural Compressor.
> Note: IPEX version >= 1.10

## Prepare

Follow [link](https://github.com/intel-innersource/frameworks.ai.models.intel-models/blob/develop/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison Jemalloc and TCMalloc.

- Install dependency
```
  conda install intel-openmp
```

- Set ENV to use AMX if you are using SPR
```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

- Install Intel® Extension for PyTorch* (IPEX)

```
  python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

> Note: Intel® Extension for PyTorch* has PyTorch version requirement. Please check more detailed information via the URL below.


## Run

### Bert-Large Inference

If IPEX version is equal or higher than 1.12, please install transformers 4.19.0. We can use the model from huggingface model hub and squad dataset from datasets package, run script `run_qa.py` with command as following.

- Install transformers
```
  pip install transformers == 4.19.0
```
- Command
```
  python run_qa.py 
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --no_cuda \
    --tune \
    --output_dir ./savedresult 

```


```
  bash run_tuning.sh --topology="bert_large_ipex"
```
```
  bash run_benchmark.sh --topology="bert_large_ipex" --mode=performance
```

### Distilbert-base Inference

For distilbert-base, the IPEX version requests equal or higher than 1.12. 

- install transformers
```
  pip install transformers == 4.19.0
```

- Command
```
  python run_qa.py 
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --no_cuda \
    --tune \
    --output_dir ./savedresult 

```


```
  bash run_tuning.sh --topology="distilbert_base_ipex"
```
```
  bash run_benchmark.sh --topology="distilbert_base_ipex" --mode=performance
```

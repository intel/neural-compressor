# Llama2

## Getting started

Please first download the data, model and preprocess the data following the steps below _within the mlperf container_. Note that if you have already downloaded the model and data prior to v4.1, you don't need to redo them. But you _need to re-run_ the preprocess_data step for the updated calibration data.

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset
```

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded pickle into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset
# to download file: open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz, open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

# unzip files
gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

# move into right directory
mv open_orca_gpt4_tokenized_llama.*.pkl build/data/llama2-70b/

```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/Llama2/Llama-2-70b-chat-hf/`,
2. preprocessed data at `build/preprocessed_data/llama2-70b/`:

- `build/preprocessed_data/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl`
- `build/preprocessed_data/llama2-70b/open_orca_gpt4_tokenized_llama.calibration_1000.pkl`

## Build and run the benchmarks

Please follow the steps below in MLPerf container.

### Prepare environments

```bash
# make sure you are in mlperf's container
bash setup.sh

```

### Quantize model

Use [Intel/auto-round](https://github.com/intel/auto-round) to make the quantization model

```bash
# quantize model
bash quantize_70b.sh

# you can also use the default quantization config of autoround
# run the `python simple_autoround.py` directly
```

### Benchmark

```bash
# inference

VLLM_USE_STANDALONE_COMPILE=0 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1,2 python main.py --mlperf-conf mlperf.conf --model-path build/models/Llama2/Llama-2-70b-chat-hf-quantized --dataset-path build/preprocessed_data/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl --tensor-parallel 2 --warmup --user-conf user.conf --accuracy --batch-size 128

# evaluation
CUDA_VISIBLE_DEVICES=2 python evaluation_scripts/evaluate-accuracy.py --checkpoint-path build/models/Llama2/Llama-2-70b-chat-hf-quantized  --mlperf-accuracy-file build/logs/mlperf_log_accuracy.json --dataset-file build/preprocessed_data/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl --dtype int64

```


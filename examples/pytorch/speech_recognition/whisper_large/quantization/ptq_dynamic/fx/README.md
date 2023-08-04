Step-by-Step
============

This document describes the step-by-step instructions for reproducing [openai/whisper-large](https://huggingface.co/openai/whisper-large) from [transformers](https://github.com/huggingface/transformers.git) tuning results with Intel® Neural Compressor.

## Prerequisite

### 1. Environment
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
```shell
cd examples/pytorch/speech_recognition/whisper_large/quantization/ptq_static/fx
pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).

```
# Run

`--dataset_location` is used to get local saved huggingface/datasets cache.

```bash
bash run_quant.sh --topology=whisper_large --output_dir=./saved_results
```

```bash
# fp32
bash run_benchmark.sh --topology=whisper_large --output_model=./saved_results --mode=performance
bash run_benchmark.sh --topology=whisper_large --output_model=./saved_results --mode=benchmark
# int8
bash run_benchmark.sh --topology=whisper_large --output_model=./saved_results --mode=performance --int8=true
bash run_benchmark.sh --topology=whisper_large --output_model=./saved_results --mode=benchmark --int8=true
```


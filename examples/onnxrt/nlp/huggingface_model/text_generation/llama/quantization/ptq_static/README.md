Step-by-Step
============

This example confirms llama's accuracy and speed based on [lambada](https://huggingface.co/datasets/lambada).

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

```bash
optimum-cli export onnx --model decapoda-research/llama-7b-hf --task text-generation-with-past ./llama_7b
optimum-cli export onnx --model decapoda-research/llama-13b-hf --task text-generation-with-past ./llama_13b
```

# Run

## 1. Quantization

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_tune \ # folder path to save onnx model
                  --batch_size=batch_size # optional \
                  --dataset NeelNanda/pile-10k \
                  --alpha 0.6 \ # 0.6 for llama-7b, 0.8 for llama-13b
                  --quant_format="QOperator" # or QDQ, optional
```

## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional 
                      --mode=accuracy \
                      --tokenizer=decapoda-research/llama-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -m 0 -C 0-3 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                          --mode=performance \
                                          --batch_size=batch_size # optional \
                                          --tokenizer=decapoda-research/llama-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                                          --intra_op_num_threads=4
```

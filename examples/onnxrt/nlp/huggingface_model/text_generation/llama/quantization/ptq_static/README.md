Step-by-Step
============

This example confirms llama's accuracy and speed based on [lambada](https://huggingface.co/datasets/lambada).

# Prerequisite

## 1. Environment
```shell
pip install neural-compressor
SKIP_RUNTIME=True pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. There are other models available that can be used for quantization. The following table shows a few models' configurations:

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | 80 | 64 | 8192 |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 80 | 64 | 8192 |

Export to ONNX model:
```bash
optimum-cli export onnx --model meta-llama/Llama-2-7b-hf --task text-generation-with-past ./Llama-2-7b-hf
```

# Run

## 1. Quantization

Run SmoothQuant

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_tune \ # folder path to save onnx model
                  --batch_size=batch_size # optional \
                  --dataset NeelNanda/pile-10k \
                  --alpha 0.6 \ # 0.6 for llama-7b, 0.8 for llama-13b
                  --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                  --quant_format="QOperator" # or QDQ, optional
```

Additionally set `--layer-wise=True` to use layer-wise quantization to save your memory. Please note that layer-wise quantization for ONNX models is still under development and only support W8A8 quantization now. More details please refer to [layer wise quantiation](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_layer_wise.md).

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_tune \ # folder path to save onnx model
                  --batch_size=batch_size # optional \
                  --dataset NeelNanda/pile-10k \
                  --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                  --quant_format="QOperator" \ # or QDQ, optional
                  --layer-wise=True
```


## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional 
                      --mode=accuracy \
                      --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -m 0 -C 0-3 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                          --mode=performance \
                                          --batch_size=batch_size # optional \
                                          --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                                          --intra_op_num_threads=4
```

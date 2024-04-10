Step-by-Step
============

This example confirms llama's weight only accuracy on [lambada](https://huggingface.co/datasets/lambada).

# Prerequisite

## 1. Environment
```shell
# build and install onnxruntime
git clone https://github.com/luoyu-intel/ort.git
cd ort
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --skip_tests --build_wheel --skip_submodule_sync --use_dnnl
pip install build/Linux/RelWithDebInfo/dist/onnxruntime_dnnl-xxx.whl

# install neural-compressor
git clone -b mengni/ns_ort https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -e .
cd examples/onnxrt/nlp/huggingface_model/text_generation/llama/quantization/weight_only

# install transformers
git clone -b v4.31.0 https://github.com/huggingface/transformers.git
cp transformer_patch transformers
cd transformers
git apply transformer_patch
pip install -e .
cd ..

# install optimum
git clone -b v1.17.1 https://github.com/huggingface/optimum.git
cp optimum_patch optimum
git apply optimum_patch
pip install -e .
cd ..

pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

> Note: Weight-only quantization in Intel® Neural Compressor is still under development. We encourage you to use the `master` branch to access the latest features.

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. There are other models available that can be used for weight-only quantization. The following table shows a few models' configurations:

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
python prepare_model.py  --input_model="meta-llama/Llama-2-7b-hf" \
                         --output_model="./llama-2-7b-hf" \
                         --task=text-generation-with-past \ # or text-generation
                         --backend=onednn \ # or mlas
```

# Run

## 1. Quantization

Set `algorithm=WOQ_TUNE` to tune weight-only quantization algorithm or specify algorithm to `RTN` or `GPTQ` or `AWQ`.

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_tune \ # folder path to save onnx model
                  --batch_size=batch_size # optional \
                  --dataset=NeelNanda/pile-10k \
                  --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                  --backend=onednn \ # or mlas
                  --algorithm=RTN # support WOQ_TUNE, RTN, AWQ, GPTQ
```

## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional 
                      --mode=accuracy \
                      --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                      --backend=onednn \ # or mlas
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -C 0-23 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                      --mode=performance \
                                      --batch_size=batch_size # optional \
                                      --backend=onednn \ # or mlas
                                      --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                                      --seqlen=1024 \
                                      --max_new_tokens=32 \
                                      --iter_num=10 \
                                      --warmup_num=3 \
                                      --intra_op_num_threads=24 
```

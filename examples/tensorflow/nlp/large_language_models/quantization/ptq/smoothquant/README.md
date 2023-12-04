Step-by-Step
============

This document is used to list steps of reproducing TensorFlow Intel® Neural Compressor quantization and smooth quantization of language models such as OPT and GPT2.

## Prerequisite

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
pip install -r requirements
```
## Run


### Basic quantization

```
python main.py --model_name_or_path <MODEL_NAME>
```

`<MODEL_NAME>` can be following:

- gpt2-medium
- facebook/opt-125m

### Smooth quant

```shell
bash run_quant.sh --input_model=<MODEL_NAME>
```

Or you can use

```
python main.py --model_name_or_path <MODEL_NAME> --sq
```

## Benchmark

### Get the FP32 performance

```shell
bash run_benchmark.sh --input_model=<MODEL_NAME>
```

### Get the INT8 performance

```shell
bash run_benchmark.sh --input_model=<MODEL_NAME> --int8=true
```


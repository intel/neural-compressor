# Step-by-Step (Deprecated)

This example load a language translation model and confirm its accuracy and speed based on [GLUE data](https://gluebenchmark.com/).

# Prerequisite

## 1. Environment

```shell
git clone -b dnnl_ep --depth 1 https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -e ./

cd examples/onnxrt/nlp/huggingface_model/text_classification/mix_precision/
pip install -r requirements.txt
```

> Note: Validated ONNX Runtime [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Supported model identifier from [huggingface.co](https://huggingface.co/):

|                Model Identifier                 |
| :---------------------------------------------: |
|          Intel/bert-base-uncased-mrpc           |
|             Intel/roberta-base-mrpc             |
|           Intel/xlm-roberta-base-mrpc           |
|            Intel/camembert-base-mrpc            |
| distilbert-base-uncased-finetuned-sst-2-english |
|         Alireza1044/albert-base-v2-sst2         |
|       Intel/MiniLM-L12-H384-uncased-mrpc        |
|     philschmid/MiniLM-L6-H384-uncased-sst2      |
|         bert-base-cased-finetuned-mrpc          |
|     Intel/electra-small-discriminator-mrpc      |
|         M-FAC/bert-mini-finetuned-mrpc          |
|           Intel/xlnet-base-cased-mrpc           |
|              Intel/bart-large-mrpc              |

```bash
python prepare_model.py --input_model=Intel/bert-base-uncased-mrpc --output_model=model
```

## 3. Prepare Dataset

Download the GLUE data with `prepare_data.sh` script.

```shell
export GLUE_DIR=/path/to/glue_data
export TASK_NAME=MRPC # or SST

bash prepare_data.sh --data_dir=$GLUE_DIR --task_name=$TASK_NAME
```

# Run

If the hardware doesn't support bf16 instruction, please set flag as below to force bf16 conversion (this way will be deprecated):

```shell
export FORCE_BF16=1
```

## 1. Only mixed precision conversion

```bash
bash run.sh --input_model=path/to/model \ # model path as *.onnx
            --output_model=path/to/model_tune \ # model path as *.onnx
```

## 2. Mixed precision conversion + accuracy evaluation

Please make sure DnnlExecutionProvider is in available providers list to execute evaluation.

```bash
bash eval.sh --input_model=path/to/model \ # model path as *.onnx
            --output_model=path/to/model_tune \ # model path as *.onnx
            --dataset_location=path/to/glue/data \
            --batch_size=batch_size \  # optional
```

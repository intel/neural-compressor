# Introduction
This example is to demonstrate the accuracy improvement introduced by [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf)
for int8 models. [Lambada](https://huggingface.co/datasets/lambada) train split is used for calibration (unless specified) and validation is used for evaluation.

# 1. Environment
```shell
cd examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/ipex/smooth_quant
pip install -r requirements.txt
```
# Run
## Basic quantization
```shell
python eval_lambada.py \
  --model_name_or_path bigscience/bloom-560m \
  --int8
```

##  Smooth quant

```shell
python eval_lambada.py \
  --model_name_or_path bigscience/bloom-560m \
  --int8 \
  --sq
```

##  Smooth quant alpha auto tuning

```shell
python eval_lambada.py \
  --model_name_or_path bigscience/bloom-560m \
  --int8 \
  --sq \
  --alpha auto
```

#### For GPT-J model, please enable the fallback_add option
```shell
python eval_lambada.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --int8 \
  --sq \
  --alpha auto \
  --fallback_add 
```



## Benchmarking 

int8 benchmarking
```shell
python benchmark.py \
  --model_name_or_path bigscience/bloom-560m \
  --int8
```

fp32 benchmarking
```shell
python benchmark.py \
  --model_name_or_path bigscience/bloom-560m 
```


# Validated Models
| Model\Last token accuracy |  FP32  | INT8 (w/o SmoothQuant) | INT8 (w/ SmoothQuant) | INT8 (w/ SmoothQuant auto tuning) |
|---------------------|:------:|:----------------------:|-----------------------|-----------------------------------|
| bigscience/bloom-560m | 65.20% |         63.44%         | 66.48% (alpha=0.5)    | 64.76%                            |
| bigscience/bloom-1b7 | 71.43% |         67.78%         | 72.56% (alpha=0.5)    | 72.58%                            |
| bigscience/bloom-3b | 73.97% |         69.99%         | 74.02% (alpha=0.5)    | 74.16%                            |
| bigscience/bloom-7b1 | 77.44% |         75.46%         | 77.02%(alpha=0.5)     | 77.45%                            |
| bigscience/bloom-176b | 84.17% |         82.13%         | 83.52% (alpha=0.6)    | -                                 |
| facebook/opt-125m   | 63.89% |         63.48%         | 63.44% (alpha=0.5)    | 64.14%                            |
| facebook/opt-1.3b   | 75.41% |         73.59%         | 70.94% (alpha=0.5)    | 74.80%                            |
| facebook/opt-2.7b   | 77.79% |         78.57%         | 78.60%(alpha=0.5)     | 78.25%                            |
| facebook/opt-6.7b   | 81.26% |         76.65%         | 81.58%(alpha=0.5)     | 81.39%                            |
| EleutherAI/gpt-j-6B | 79.17% |         78.82%         | 78.84%(alpha=0.6)     | 79.29%                            |

Please note we are using the last token accuracy, just the same as the one in the repo of the official SmoothQuant code. We will change to LM-eval-harness evaluation later.

For bloom-560m, please use --kl to enable kl calibration or keep tuning to the 4th time.

For gpt-j-6B, please fallback 'add' op to fp32 manually or keep tuning.

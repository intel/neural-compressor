# Introduction
This example is to demonstrate the accuracy improvement introduced by SmoothQuant[1] for int8 models. Lambada train split is used for calibration(unless specified) and validation  is used for evaluation.

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
| Models\Accuracy       |  FP32  |    INT8 (w/o SmoothQuant)     | INT8 (w/ SmoothQuant)           |
|------------------|:------:|:-----------:|--------------|
| bigscience/bloom-560m | 65.16% | 64.96%  | 66.52% (alpha=0.5)   |
| bigscience/bloom-1b7 | 71.55% |   67.61%    | 72.81% (alpha=0.5)       |
| bigscience/bloom-3b | 74.06% |   70.73%    | 74.41% (alpha=0.5)       |
| bigscience/bloom-7b1 | 77.59% |   76.28%    | 77.18% (alpha=0.5)       |
| bigscience/bloom-176b | 84.17% |   82.13%    | 83.52% (alpha=0.6) |
| facebook/opt-125m | 63.89% |   63.54%    | 63.91% (alpha=0.5)       |
| facebook/opt-1.3b | 75.42% | 73.86% | 74.64% (alpha=0.5)  |
| facebook/opt-2.7b | 77.90% |   78.99%    | 78.91% (alpha=0.5)       |
| facebook/opt-6.7b | 81.51% |   79.44%    | 81.58% (alpha=0.5)       |



# Reference


```bibtex
@article{[xiao2022smoothquant](https://arxiv.org/pdf/2211.05100.pdf),
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Demouth, Julien and Han, Song},
  journal={arXiv},
  year={2022}
}
```

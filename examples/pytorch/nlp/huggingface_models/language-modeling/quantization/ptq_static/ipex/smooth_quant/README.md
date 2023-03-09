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
python eval_lambada.py \
  --model_name_or_path bigscience/bloom-560m \
  --int8\
  --benchmark
```

fp32 benchmarking
```shell
python eval_lambada.py \
  --model_name_or_path bigscience/bloom-560m \
  --benchmark
```


# Validated Models
| Models\Acc       |  FP32  |    int8     | SQ           |
|------------------|:------:|:-----------:|--------------|
| bigscience/bloom-560m | 0.6516 | 0.6496(kl)  | 0.6652(kl)   |
| bigscience/bloom-1b7 | 0.7155 |   0.6761    | 0.7281       |
| bigscience/bloom-3b | 0.7406 |   0.7073    | 0.7441       |
| bigscience/bloom-7b1 | 0.7759 |   0.7628    | 0.7718       |
| bigscience/bloom | 0.8417 |   0.8213    | 0.8352(alpha=0.6) |
| facebook/opt-125m | 0.6389 |   0.6354    | 0.6391       |
| facebook/opt-1.3b | 0.7542 | 0.7386(val) | 0.7464(val)  |
| facebook/opt-2.7b | 0.7790 |   0.7899    | 0.7891       |
| facebook/opt-6.7b | 0.8151 |   0.7944    | 0.8158       |



# Reference


```bibtex
@article{xiao2022smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Demouth, Julien and Han, Song},
  journal={arXiv},
  year={2022}
}
```

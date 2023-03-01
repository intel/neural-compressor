# Introduction
This example is to demonstrate the accuracy improvement introduced by SmoothQuant[1] for int8 models. Lambada train split is used for calibration(unless specified) and validation  is used for evaluation.

# 1. Environment
```shell
pip3 install torch transformers datasets
python -m pip install intel_extension_for_pytorch

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
# Validated Models
| Models\Acc |  FP32  |    int8     | SQ                |
|------------|:------:|:-----------:|-------------------|
| Bloom-560m | 0.6516 | 0.6496(kl)  | 0.6652(kl)        |
| Bloom-1.7B | 0.7155 |   0.6761    | 0.7281            |
| Bloom-3B   | 0.7406 |   0.7073    | 0.7441            |
| Bloom-7.1B | 0.7759 |   0.7628    | 0.7718            |
| Bloom-176B | 0.8417 |   0.8213    | 0.8352(alpha=0.6) |
| OPT-125M   | 0.6389 | 0.6361(val) | 0.6406(val)       |
| OPT-1.3B   | 0.7542 | 0.7386(val) | 0.7464(val)       |
| OPT-2.7B   | 0.7790 |      -      | 0.7891            |
| OPT-6.7B   | 0.8151 |      -      | 0.8158            |



# Reference


```bibtex
@article{xiao2022smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Demouth, Julien and Han, Song},
  journal={arXiv},
  year={2022}
}
```

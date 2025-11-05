PyTorch Smooth Quantization
========================================

1. [Introduction](#Introduction)
2. [Usage](#Usage)
3. [Validated Models](#Validated-Models)
4. [Supported Framework Matrix](#Supported-Framework-Matrix)


## Introduction
Quantization is a common compression operation to reduce memory and accelerate inference by converting the floating point matrix to an integer matrix. For large language models (LLMs) with gigantic parameters, the systematic outliers make quantification of activations difficult.  [SmoothQuant](https://arxiv.org/abs/2211.10438), a training free post-training quantization (PTQ) solution, offline migrates this difficulty from activations to weights with a mathematically equivalent transformation.


## Usage
### Fixed Alpha
To set a fixed alpha for the entire model, users can follow this example:

```python
from neural_compressor.torch.quantization import SmoothQuantConfig, convert, prepare


def run_fn(model):
    model(example_inputs)


quant_config = SmoothQuantConfig(alpha=0.5)
prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```
`SmoothQuantConfig` description:

`alpha`: a smooth factor to calculate the conversion per-channel scale and balance the quantization difficulty of activation and weight. Float value, default is 0.5.

> **Note:** Alpha="auto" and alpha auto-tuning was supported in old API, please stay tuned for the new API's support for auto alpha.

### Specify Quantization Rules
Intel(R) Neural Compressor support specify quantization rules by operator type for Smooth Quantization. Users can use `set_local` to fallback op type in `SmoothQuantConfig` to achieve the above purpose.

Here we don't quantize `Linear` layers.
```python
# fallback by op_type
quant_config.set_local("Linear", SmoothQuantConfig(w_dtype="fp32", act_dtype="fp32"))
prepared_model = prepare(model, quant_config=quant_config, example_inputs=example_inputs)
run_fn(prepared_model)
q_model = convert(prepared_model)
```

To get more information, please refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/smooth_quant).


## Validated Models
Neural Compressor: 2.1

IPEX (Intel Extension for PyTorch): 2.0/2.1

Dataset: lambada_openai

Task: text-generation provided by [ITREX](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/text-generation/quantization)

alpha [0.4, 0.6] is sweet spot region in SmoothQuant paper.

A list of models that achieved a <1% accuracy drop is shown below.

| Model/Last token accuracy |  FP32 Accuracy   | INT8 (w/ SmoothQuant) | Notes |
|:----------:|:------:|:------:|-----------------------------------|
| bigscience/bloom-560m | 0.354 | 0.3542 | alpha=0.5, Ipex 2.1 |
| bigscience/bloom-1b7  | 0.4634 | 0.4936 | alpha=0.5, Ipex 2.0 |
| bigscience/bloom-3b   | 0.518 | 0.5185 | alpha=0.8, Ipex 2.1 | 
| bigscience/bloom-7b1  | 0.5764 | 0.5977 | alpha=0.5, Ipex 2.0 |
| bigscience/bloomz-560m  | 0.3947 | 0.3930 | alpha=0.8, Ipex 2.1 |
| bigscience/bloomz-1b7  | 0.4828 | 0.4906 | alpha=0.5, Ipex 2.1 |
| bigscience/bloomz-3b   | 0.5018 | 0.4980 | alpha=0.5, Ipex 2.1 | 
| bigscience/bloomz-7b1  | 0.5593 | 0.5552 | alpha=0.5, Ipex 2.1 |
| facebook/opt-125m   | 0.379 | 0.3757 | alpha=0.5, Ipex 2.1 |
| facebook/opt-350m   | 0.4516 | 0.4533 | alpha=0.8, Ipex 2.1 |
| facebook/opt-1.3b   | 0.5789 | 0.5742 | alpha=0.8, Ipex 2.0 |
| facebook/opt-2.7b   | 0.6365 | 0.6404 | alpha=0.5, Ipex 2.0 |
| facebook/opt-6.7b   | 0.6769 | 0.6804 | alpha=0.5, Ipex 2.0 |
| facebook/opt-13b   | 0.6872 | 0.6814 | alpha=0.5, Ipex 2.1 |
| facebook/opt-30b   | 0.7149 | 0.7128 | alpha=0.5, Ipex 2.1 |
| facebook/opt-66b   | 0.7398 | 0.7326 | alpha=0.5, Ipex 2.1 |       
| LLaMa-7b | 0.7361 | 0.7357 | alpha=0.8, Ipex 2.1 |
| LLaMa-13b | 0.7627 | 0.7590 | alpha=0.7, Ipex 2.1 |
| LLaMa-30b | 0.7759 | 0.7840 | alpha=0.7, Ipex 2.1 |
| LLaMa-65b | 0.7908 | 0.7957 | alpha=0.9, Ipex 2.1 |
| EleutherAI/gpt-j-6B* | 0.6831 | 0.6821 | alpha=1.0, Ipex 2.1 |
| MBZUAI/LaMini-GPT-124m | 0.3804 | 0.3887 | alpha=0.5, Ipex 2.1 |
| MBZUAI/LaMini-GPT-774m | 0.5048 | 0.5057 | alpha=0.5, Ipex 2.1 |
| MBZUAI/LaMini-GPT-1.5b | 0.5443 | 0.5436 | alpha=0.5, Ipex 2.1 |
| mosaicml/mpt-7b-chat | 0.655 | 0.6499 | alpha=0.7, Ipex 2.1 |
| stabilityai/stablelm-base-alpha-3b | 0.4172 | 0.4149 | alpha=0.6, Ipex 2.1 |
| togethercomputer/RedPajama-INCITE-Base-3B-v1 | 0.6542 | 0.6735 | alpha=0.5, Ipex 2.1 |
| togethercomputer/RedPajama-INCITE-Chat-3B-v1* | 0.6718 | 0.6740 | alpha=0.5, Ipex 2.0 |
| togethercomputer/RedPajama-INCITE-Instruct-3B-v1* | 0.6569 | 0.6621 | alpha=0.5, Ipex 2.0 |
| togethercomputer/RedPajama-INCITE-Base-7B-v0.1* | 0.7143 | 0.7221 | alpha=0.5, Ipex 2.0 |
| togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1* | 0.6895 | 0.6953 | alpha=0.5, Ipex 2.0 |
| databricks/dolly-v1-6b* | 0.6866 | 0.6895 | alpha=0.8, Ipex 2.1 |
| databricks/dolly-v2-3b* | 0.6297 | 0.6247 | alpha=0.5, Ipex 2.1 |
| tiiuae/falcon-7b-instruct | 0.6437 | 0.6392 | alpha=0.7, Pytorch |

Please note that for models with asterisk(*), we have set all add ops to FP32 during quantization step to achieve desirable results.


## Supported Framework Matrix

| Framework | Alpha        | Folding    |
|:---------:|--------------|------------|
| PyTorch   | [0-1] | False      |
| IPEX      | [0-1] | True / False(Version>2.1) |

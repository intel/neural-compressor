# Smooth Quant

1. [Introduction](#Introduction)
2. [Quantization Fundamentals](#Quantization-Fundamentals)
3. [SmoothQuant and Our Enhancement](#SmoothQuant-and-Our-Enhancement)
4. [Usage](#Usage)
5. [Reference](#reference)


## Introduction

Quantization is a common compression operation to reduce memory and accelerate inference by converting the floating point matrix to an integer matrix. For large language models (LLMs) with gigantic parameters, the systematic outliers make quantification of activations difficult.  [SmoothQuant](https://arxiv.org/abs/2211.10438), a training free post-training quantization (PTQ) solution, offline migrates this difficulty from activations to weights with a mathematically equivalent transformation.

## Quantization Fundamentals

Quantization is a common compression operation to reduce memory and accelerate inference; therefore, the difficulty of LLM deployment can be alleviated. Quantization converts the floating point matrix to an integer matrix.

The equation of quantization is as follows:

$$
X_{int8} = round(X_{fp32}/S) + Z \tag{1}
$$

where $X_{fp32}$ is the input matrix, $S$ is the scale factor,  $Z$ is the integer zero point.

### Per-tensor & Per-channel

There are several choices of sharing quantization parameters among tensor elements, also called quantization granularity. The coarsest level, per-tensor granularity, is that all elements in the tensor share the same quantization parameters. Finer granularity means sharing quantization parameters per row or per column for 2D matrices and per channel for 3D matrices. Similarly, the finest granularity is that each element has an individual set of quantization parameters.


However, due to the model accuracy and computational consumption, per-tensor or per-channel are usually adopted. **Through mathematical calculations, per-channel could bring lower quantization loss but has some limitations, that is why normally we use per-channel for weight quantization and per-tensor for activation/input quantization**

#### Per-channel limitation

Though per-channel quantization could bring lower quantization error, we could not apply it for activations due to the difficulty of the dequantization. We would prove it in the following image and the zero point of quantization would be ignored for simplicity.

The image on the left presents a normal linear forward  with 1x2 input $x$ and 2x2 weight $w$. The results $y$ could be easily obtained by simple mathematics. In the middle image, we apply per-tensor quantization for activations and per-channel quantization for weights; the results after quantization that are denoted by $y_1$ and $y_2$, could be easily dequantized to the float results $y_{fp1}$ and $y_{fp2}$ by per channel scale $1.0/s_1s_x$ and $1.0/s_2s_x$. However, after applying per-channel quantization for activations (right image), we could not dequantize the  $y_1$ and  $y_2$ to float results.

<div align="center">
    <img src="../../source/imgs/sq_pc.png"/>
</div>


## SmoothQuant and Our Enhancement

### SmoothQuant

In the previous subsection, we have explained why per-channel quantization could not be applied for s, even though it could lead to lower quantization loss. However, the quantization error loss of activations plays an important role in the accuracy loss of model quantization[^2][^3][^4]. 



To reduce the quantization loss of activations, lots of methods have been proposed. In the following, we briefly introduce SPIQ[^2], Outlier Suppression[^3] and Smoothquant[^4]. All these three methods share a similar idea to migrate the difficulty from activation quantization to weight quantization but differ in how much difficulty to be transferred.


So **the first question is how to migrate the difficulty from activation to weights?** The solution is straightforward, that is to convert the network to an output equivalent network that is presented in the image below and apply quantization to this equivalent network. The intuition is that each channel of activations could be scaled to make it more quantization-friendly, similar to a fake per-channel activation quantization.

<div align="center">
    <img src="../../source/imgs/sq_convert.png"/>
</div>


Please note that this conversion will make the quantization of weights more difficult, because the scales attached to weights shown above are per-input-channel, while quantization of weights is per-output-channel or per-tensor.

So **the second question is how much difficulty to be migrated**, that is how to choose the **conversion per-channel scale** $s_{x1}$ and $s_{x2}$ from the above image. Different works adopt different ways.

*SPIQ* just adopts the quantization scale of activations as the conversion per-channel scale.

*Outlier suppression* adopts the scale of the preceding layernorm as the conversion per-channel scale.

*Smoothquant* introduces a hyperparameter $\alpha$ as a smooth factor to calculate the conversion per-channel scale and balance the quantization difficulty of activations and weights.

$$
s_j = max(|X_j|)^\alpha/max(|W_j|)^{1-\alpha} \tag{4}
$$

j is the index of the input channels.



<div align="center">
    <img src="../../source/imgs/smoothquant.png" height="250"/>
</div>



For most of the models such as OPT and BLOOM, $\alpha = 0.5$ is a well-balanced value to split the difficulty of weight and activation quantization. A larger $\alpha$ value could be used on models with more significant activation outliers to migrate more quantization difficulty to weights.


### Our enhancement: 

#### Algorithm: Auto-tuning of $\alpha$.

SmoothQuant method aims to split the quantization difficulty of weights and activations by using a fixed-value $\alpha$ for an entire model. However, as the distributions of activation outliers vary not only across different models but also across different layers within a model, we hereby propose a method to obtain layer-wise optimal $\alpha$ values with the ability to tune automatically.

Our proposed method consists of 7 major steps:

-    Calculate input minimum and maximum values of operators to be smoothed.
-    Find a list of operators on which smoothquant could be performed.
-    Set a $\alpha$ value based on user-defined $\alpha$ values.
-    Calculate smoothing factor using the current $\alpha$ value, adjust parameters accordingly and forward the adjusted model given an input sample.
-    Perform per-channel quantization_dequantization of weights and per-tensor quantization_dequantization of activations to predict output.
-    Calculate the accuracy loss with respect to FP32 output, iterate the previous three steps given each $\alpha$ value and save the loss per alpha.
-    Stop iterating if the maximum times of trial is reached and output the quantized model with a minimum accuracy loss.



Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value of an input LayerNorm op of a transformer block. Both alpha range and criterion could be configured in auto_alpha_args.

In our experiments, an $\alpha$ range of [0.0, 1.0] with a step_size of 0.1 is found to be well-balanced one for the majority of models.


## Usage
There are two ways to apply smooth quantization: 1) using a fixed `alpha` for the entire model or 2) determining the `alpha` through auto-tuning.

### Using a fixed `alpha`
To set a fixed alpha for the entire model, users can follow this example:

```python
from neural_compressor.tensorflow import SmoothQuantConfig, StaticQuantConfig

quant_config = [SmoothQuantConfig(alpha=0.5), StaticQuantConfig()]
q_model = quantize_model(
    output_graph_def, 
    [sq_config, static_config], 
    calib_dataloader
)
```
The `SmoothQuantConfig` should be combined with `StaticQuantConfig` in a list because we still need to insert QDQ and apply pattern fusion after the smoothing process.


### Determining the `alpha` through auto-tuning
Users can search for the best `alpha`  for the entire model.The tuning process looks for the optimal `alpha` value from a list of `alpha` values provided by the user.

Here is an example:

```python
from neural_compressor.tensorflow import StaticQuantConfig, SmoothQuantConfig

custom_tune_config = TuningConfig(
    config_set=[SmoothQuantConfig(alpha=[0.5, 0.6, 0.7]), StaticQuantConfig()]
)
best_model = autotune(
    model="fp32_model",
    tune_config=custom_tune_config,
    eval_fn=eval_fn_wrapper,
    calib_dataloader=calib_dataloader,
)
```
> Please note that, it may a considerable amount of time as the tuning process applies each `alpha` to the entire model and uses the evaluation result on the entire dataset as the metric to determine the best `alpha`.

## Reference

[^1]: Jason, Wei, et al. "Emergent Abilities of Large Language Models". Published in Transactions on Machine Learning Research (2022).


[^2]: Yvinec, Edouard, et al. "SPIQ: Data-Free Per-Channel Static Input Quantization." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.


[^3]: Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).


[^4]: Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022).

# Smooth Quant

1. [Introduction](#Introduction)
2. [Quantization Fundamentals](#Quantization-Fundamentals)
3. [SmoothQuant and Our Enhancement](#SmoothQuant-and-Our-Enhancement)
4. [Validated Models](#Validated-Models)
5. [Example](#Example)



## Introduction

Quantization is a common compression operation to reduce memory and accelerate inference by converting the floating point matrix to an integer matrix. For large language models (LLMs) with gigantic parameters, the systematic outliers make quantification of activations difficult.  [SmoothQuant](https://arxiv.org/abs/2211.10438), a training free post-training quantization (PTQ) solution, offline migrates this difficulty from activations to weights with a mathematically equivalent transformation.

## Quantization Fundamentals

Quantization is a common compression operation to reduce memory and accelerate inference; therefore, the difficulty of LLM deployment can be alleviated. Quantization converts the floating point matrix to an integer matrix.

The equation of quantization is as follows:

$$
X_{int8} = round(X_{fp32}/S) + Z \tag{1}
$$

where $X_{fp32}$ is the input matrix, $S$ is the scale factor,  $Z$ is the integer zero point.

### Per-tenor & Per-channel

There are several choices of sharing quantization parameters among tensor elements, also called quantization granularity. The coarsest level, per-tensor granularity, is that all elements in the tensor share the same quantization parameters. Finer granularity means sharing quantization parameters per row or per column for 2D matrices and per channel for 3D matrices. Similarly, the finest granularity is that each element has an individual parameter.

However, due to the model accuracy and computational consumption, per-tensor or per-channel are usually adopted. **In the following part, We will show per-channel could bring lower quantization loss but with some limitations, that is why normally we use per-channel for weight quantization and per-tensor for activation/input quantization**

#### Per-tensor example

Suppose the weight tensor is：

```python
import torch
W = torch.Tensor(
    [[0.6839, 0.4741, 0.7451],
    [0.9301, 0.1742, 0.6835]]
    )
```

According to the formula (1), we need to scale $S$ and zero point $Z$ to calculate the integer matrix.

$$
S = \frac{X_{max} - X{min}}{2^b -1} \tag{2}
$$

$$
Z = -round(X_{min/}/S) \tag{3}
$$

The per-tensor quantization function is:

```python
def quantize(x, num_bits=8):
    q_min, q_max = 0, 2. ** num_bits - 1.
    scale = (torch.max(x) - torch.min(x)) / (2 ** num_bits - 1)
    scale = torch.clip(scale, min=1e-5)
    bias = torch.round(0 - (torch.min(x)) / scale)
    q_x = x / scale + bias
    q_x.clamp_(q_min, q_max).round_()
    print(f'scale = {scale}, bias = {bias}')
    return q_x
```

Then we can get the quantized $W_{q}$:

```bash
>>> W_q = quantize(W)
scale = 0.00296431384049356, bias = -59.0
>>> W_q
tensor([[172., 101., 192.],
        [255.,   0., 172.]])
```

With the value of scale and bias, we can dequantize the tensor.

```python
def dequantize(q_x, scale, bias):
    return scale * (q_x - bias)
```

```bash
>>> W_dq = dequantize(W_dq, 0.001, -50)
>>> W_dq
tensor([[0.1220, 0.0500, 0.1430],
        [0.2570, 0.0500, 0.1890]])
>>> loss = torch.nn.MSELoss()(W_dq, W)
>>> loss.item()
0.1983354538679123

>>> W_dq = dequantize(W_q, 0.0020850980654358864, -70)
>>> W_dq
tensor([[0.6848, 0.4743, 0.7440],
        [0.9308, 0.1749, 0.6848]])
>>> loss = torch.nn.MSELoss()(W_dq, W)
>>> loss.item()

```

The difference between $W$ and $W_{dq}$ shows that quantization affects precision and appropriate values of scale and zero point will reduce the loss of precision. 

#### Per-channel example

Similarly, the example of per-channel quantization is as follows:

```python
def quantize_per_channel(x, num_bits=8):
    q_min, q_max = 0, 2. ** num_bits - 1.
    x_tmp = x.detach().reshape(x.shape[0], -1)
    scales = x_tmp.max(dim=-1, keepdim=True)[0] / (2 ** num_bits - 1)
    bias =  torch.round(0 - x_tmp.min(dim=-1, keepdim=True)[0].divide(scales))
    q_x = x_tmp.divide(scales) + bias
    q_x.clamp_(q_min, q_max).round_()
    print(f'scale = {scales}, \nbias = {bias}')
    return q_x

def dequantize_per_channel(q_x, scales, bias):
    print(q_x, scales, bias)
    print(scales * (q_x - bias))
    return scales * (q_x - bias)
```

```bash
>>>W_q = quantize_per_channel(W)
scale = tensor([[0.0029],
        [0.0036]]), 
bias = tensor([[-162.],
        [ -48.]])
>>>W_q
tensor([[ 72.,   0.,  93.],
        [207.,   0., 139.]])

>>>scales = torch.tensor([[0.0027],[0.0017]])
>>>bias = torch.tensor([[-66.],[-87.]])
>>>W_dq = dequantize_per_channel(W_q, scales, bias)
>>>W_dq
tensor([[0.6837, 0.4734, 0.7451],
        [0.9301, 0.1751, 0.6821]])
```

And the loss is

```bash
>>> loss = torch.nn.MSELoss()(W_dq, W)
>>> loss.item()
5.637690492221736e-07
```

Through this example, we can see that per-channel quantization has finer granularity and has lower loss.

#### Matmul quantization example

For a linear layer in most model, $Y=X \cdot W$, we can quantize both the weights and activations in order to reduce the storage and accelerate inference.
Using per-tensor scale quantization to show the process.

```python
def quantize_per_tensor_absmax(x, n_bits=8):
    scales = x.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x

def dequantize(q_x, scale):
    return scale * q_x
```

Random initialize the $W$ and $Y$, then calculate the result of $Y=X \cdot W$

```bash
>>>W = torch.rand(2, 3, dtype=torch.float32)
>>>X = torch.rand(3, 4, dtype=torch.float32)
>>>W
tensor([[0.0806, 0.7589, 0.6038],
        [0.3815, 0.5040, 0.7174]])
>>>X
tensor([[0.5444, 0.5826, 0.7772, 0.5555],
        [0.3740, 0.3253, 0.0698, 0.1381],
        [0.5972, 0.0086, 0.0737, 0.8298]])
>>>Y = torch.matmul(W, X)
>>>Y
tensor([[0.6883, 0.2991, 0.1601, 0.6506],
        [0.8246, 0.3924, 0.3845, 0.8768]])
```

Quantize weight and activation, matmul(quantize(X), quantize(Y))

```bash
>>>W_q, W_scale = quantize_per_tensor_absmax(W)
>>>X_q, X_scale = quantize_per_tensor_absmax(X)
>>>print(f'{W_q}\n{W_scale.item()}')
>>>print(f'{X_q}\n{X_scale.item()}')
tensor([[ 13., 127., 101.],
        [ 64.,  84., 120.]])
0.0059755356051027775
tensor([[ 83.,  89., 119.,  85.],
        [ 57.,  50.,  11.,  21.],
        [ 91.,   1.,  11., 127.]])
0.006533813662827015

>>>Y_q = torch.matmul(W_q, X_q)
>>>Y_q
tensor([[17509.,  7608.,  4055., 16599.],
        [21020., 10016.,  9860., 22444.]])
>>>Y_dq = dequantize(Y, W_scale * X_scale)
>>>Y_dq
tensor([[0.6836, 0.2970, 0.1583, 0.6481],
        [0.8207, 0.3911, 0.3850, 0.8763]])
```

#### Per-channel limitation

Though per-channel quantization could bring lower quantization error, we could not apply it for activations due to the difficulty of the dequantization. We would prove it in the following image and the zero point of quantization would be ignored for simplicity.

The left side of the image presents a normal linear forward  with 1x2 input $x$ and 2x2 weight $w$. The results $y$ could be easily obtained by simple mathematics. In the middle sub-image, we apply per-tensor quantization for activations and per-channel quantization for weights; the results after quantization that are denoted by $y_1$ and $y_2$, could be easily dequantized to the float results $y_{fp1}$ and $y_{fp2}$ by per channel scale $1.0/s_1s_x$ and $1.0/s_2s_x$. However, after applying per-channel quantization for activation on the right sub-image, we could not dequantize the  $y_1$ and  $y_2$ to float results.

<div align="center">
    <img src="./imgs/sq_pc.png"/>
</div>


## SmoothQuant and Our Enhancement

### SmoothQuant

In the previous subsection, we have explained why per-channel quantization could not be applied for activation, even though it could lead to lower quantization loss. However, the quantization error loss of activation plays an important role in the accuracy loss of model quantization[^2][^3][^4]. 



To reduce the quantization loss of activations, lots of methods have been proposed. In the following, we briefly introduce SPIQ[^2], Outlier Suppression[^3] and Smoothquant[^4]. All these three methods share a similar idea to migrate the difficulty from activation quantization to weight quantization but differ in how much difficulty to be transferred.


So **the first question is how to migrate the difficulty from activation to weights?** The solution is straightforward, that is to convert the network to an output equivalent network that is presented in the image below and apply quantization to this equivalent network. The intuition is that each channel of activation could be scaled to make it more quantization-friendly, similar to a fake per-channel activation quantization.

<div align="center">
    <img src="./imgs/sq_convert.png"/>
</div>


Please note that this conversion will make the quantization of weights more difficult, because the scales attached to weights showed above are per-input-channel, while quantization of weights is per-output-channel or per-tensor.

So **the second question is how much difficulty to be migrated**, that is how to choose the **convention per-channel scale** $s_{x1}$ and $s_{x2}$ on the above image. Different works adopt different ways.

*SPIQ* just adopts the quantization scale of activations as the convention per-channel scale.

*Outlier suppression* adopts the scale of the preceding layernorm as the convention per-channel scale.

*Smoothquant* introduces a hyperparameter $\alpha$ as a smooth factor to calculate the convention per-channel scale and balance the quantization difficulty of activation and weight.

$$
s_j = max(|X_j|)^\alpha/max(|W_j|)^{1-\alpha} \tag{4}
$$

j is the index of the input channels.



<div align="center">
    <img src="./imgs/smoothquant.png" height="250"/>
</div>



For most of the models such as OPT and BLOOM, $\alpha = 0.5$ is a well-balanced value to split the difficulty of weight and activation quantization. A larger $\alpha$ value could be used on models with more significant activation outliers to migrate more quantization difficulty to weights.


### Our enhancement: 

#### Algorithm: Layer-wise Auto-tuning of $\alpha$.

SmoothQuant method aims to split the quantization difficulty of weight and activation by using a fixed-value $\alpha$ for an entire model. However, as the distributions of activation outliers vary not only across different models but also across different layers within a model, we hereby propose a method to obtain layer-wise optimal $\alpha$ values with the ability to tune automatically.

Our proposed method consists of 5 major steps:

-    Hook input and output values of all layers using register_forward_hook.
-    Generate a list of $\alpha$ values given user-defined $\alpha$ range and step_sizes.
-    Recalculate smoothing factor given an $\alpha$ value and adjust parameters(weights and activations).
-    Perform per-channel quantization_dequantization of weights and per-tensor quantization_dequantization of inputs to predict the layer-wise output corresponding to the given $\alpha$ value.
-    Calculate the mean-squared loss with respect to the actual output value, recover the adjusted parameters and save the layer-wise optimal $\alpha$ values.



Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value of an input LayerNorm op of a transformer block.

In our experiments, an $\alpha$ range of [0.3, 0.7] with a step_size of 0.05 is found to be well-balanced one for the majority of models.

#### Engineering 

*fully automated*: the user only needs to pass a model and dataloader

```python
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
sq = TorchSmoothQuant(model, dataloader)
sq.transform(alpha) ##alpha could be a float or a string ’auto‘
```

please note that we rely on torch jit to analyze the model. If you are using huggingface model, you could set torchscript to True when loading the model or set the return_dict to False"

*support lots of fusing patterns*: the official code only supports 

```bash
linear->layernorm
```

while we support the following patterns

```bash
conv2d/linear->relu/leakyrelu/hardtanh->conv2d/linear/layernorm/batchnorm/instancenorm/t5norm/llamanorm/groupnorm/

conv2d/linear->conv2d/linear/layernorm/batchnorm/instancenorm/t5norm/llamanorm/groupnorm
```

For opt models, we could fuse one more layer than the official code, because the fc2 layer in the block follows the linear->relu->linear pattern.

## Validated Models
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


## Example

User could refer to [examples](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/ipex/smooth_quant/README.md) on how to use smooth quant.


## Reference

[^1]: Jason, Wei, et al. "Emergent Abilities of Large Language Models". Published in Transactions on Machine Learning Research (2022)

[^2]: Yvinec, Edouard, et al. "SPIQ: Data-Free Per-Channel Static Input Quantization." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

[^3]: Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).

[^4]: Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022)..
# SmoothQunat

## INC introduction

Intel® Neural Compressor(INC) is an  open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime, and MXNet). INC aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/), and [MXNet](https://mxnet.apache.org/), as well as Intel extensions such as [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch). In addition, the tool showcases the key features, typical examples, and broad collaborations as below:

- Support a wide range of Intel hardware such as [Intel Xeon Scalable processor](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html), [Intel Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html), and [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) with extensive testing; support AMD CPU, ARM CPU, and NVidia GPU through ONNX Runtime with limited testing
- Validate more than 10,000 models such as [Stable Diffusion](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/text-to-image/quantization), [GPT-J](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx), [BERT-Large](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx), and [ResNet50](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx) from popular model hubs such as [Hugging Face](https://huggingface.co/), [Torch Vision](https://pytorch.org/vision/stable/index.html), and [ONNX Model Zoo](https://github.com/onnx/models#models), by leveraging zero-code optimization solution [Neural Coder](https://github.com/intel/neural-compressor/blob/master/neural_coder#what-do-we-offer) and automatic [accuracy-driven](https://github.com/intel/neural-compressor/blob/master/docs/source/design.md#workflow) quantization strategies
- Collaborate with cloud marketplace such as [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/bitnami-launchpad/inc-tensorflow-intel?project=verdant-sensor-286207), [Amazon Web Services](https://aws.amazon.com/marketplace/pp/prodview-yjyh2xmggbmga#pdp-support), and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/bitnami.inc-tensorflow-intel), software platforms such as [Alibaba Cloud](https://www.intel.com/content/www/us/en/developer/articles/technical/quantize-ai-by-oneapi-analytics-on-alibaba-cloud.html) and [Tencent TACO](https://new.qq.com/rain/a/20221202A00B9S00), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [PyTorch](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html), [ONNX](https://github.com/onnx/models#models), and [Lightning AI](https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/advanced/post_training_quantization.rst)

**Visit the Intel® Neural Compressor online document website at: https://intel.github.io/neural-compressor.**

## LLM 
### introduction
A Large language mode (LLM) is a language model with billions of weights or more, trained on massive data to solve natural language processing (NLP) and natural language generation (NLG) tasks. Base on large amount of training text such as wikipedia and corpora, LLMs can knowledge about the structure of sentence, the relationship between words and the meaning of whole documents. More complex network structures and more parameters provide LLMs ability to face the complexity and polysemy of natural language.

LLMs are used in a wide variety of applications. They can be used to generate text, such as chatbots and virtual assistants, or fine tuned with a task-specific training for application to downstream tasks, like machine translation, emotion analysis, text classification, fraud detection and etc. 

### deployment challenges
LLMs show excellent performance in many tasks, and research shows that LLMs with bigger number of parmaters can have better performance.

![](./imgs/model_scale_accuracy.png)

Therefore, the scale of LLMs grows exponentially. For example, GPT-2, released in 2019, has 1.5 billion parameters and the number of parameters increase to 175 billions when GPT-3 released in 2020.

Billions or more paramaters make LLMs perform well in various tasks, howerever, also make it more difficult to deploy. Models are usually loaded on servers which have limited memory for infering tasks. The large scale of LLM make the process of inference very slow and even worse, it cannot work if the infrastructure does not meet the requirement.
## Quantization Fundamentals

Quantization is a common compression operation to reduce memory and accelerate inference, therefore, the difficulty of LLM deployment can be alleviate. Quantization convert the floating point matrix to an integer matrix.  `Affine quantization` and `Scale quantization`, also called `asymmetric quantization` and `symmetric quantization`, are two common range mapping techniques used in tensor conversion between different data types.
The math equation of quantization is like:

$$
X_{int8} = round(X_{fp32}/S) + Z \tag{1}
$$

where $X_{fp32}$ is the input matrix, $S$ is the scale factor,  $Z$ is the integer zero point.

### Granularity
There are several choices of sharing quantization parameters among tensor elements, also called quantization granularity. The coarest level, per-tensor granularity, is that all elements in the tensor share the same quantization parameters. Finer granularity shared quantization parameters per row or per column for 2D matrics and per channel for 3D matrics. Similarly, each element has individual parameters is the finest granularity. 

However, considering the model accuracy and computational consumption, we use per-tensor or per-channel for weight quantization and per-tensor for activation quantization.

### example
We will through the example to show how quantization works.

Suppose the weight tensor is：
```python
import torch
W = torch.Tensor(
    [[0.6839, 0.4741, 0.7451],
    [0.9301, 0.1742, 0.6835]]
    )
```
As the formula (1) showed, we need scale $S$ and zero point $Z$ to calculate the integer matrix.
$$
S = \frac{X_{max} - X{min}}{2^b -1}\\
Z = -round(X_{min/}/S)
$$
 Therefore the per-tensor quantization function is:
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
Then we can get the quantized $W$:
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
The difference between $W$ and $W_{dq}$ shows that quantization affects precision and choose appropriate value of scale and zero point will reduce the loss of precision. 

Similary, the example of per-channel quantization as:
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
>>>bias = torch.tensor([[-66.],[-87.]]
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
Through this example, we can see that per-channel quantization is finer granularity and has lower loss.

### Weights and Activations
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

**TODO**

1 per tensor add example(guoheng)

2 add per-channel code example to show low loss (guoheng)

3 matmul activation and weight quant(heng)

4 matmul example to show the activation can't be quantized with per channel (wenhua), normally we use per-channel for weight quantization and per-tensor for activation quantization

5 to show activation quantization loss is important to some models(heng)



## SmoothQuant and our enhancement

**TODO(wenhua)** introduce some former work with similar idea
[202302-arxiv]Outlier Suppression Pushing the Limit of Low-bit Transformer Language Models
SPIQ: Data-Free Per-Channel Static Input Quantization



For LLMs, activations are much harder to quantize than weights due to the outliers. The activation variance is large amongst the channels for a given token but is small between magnitudes of a given channels. Therefore, the quantization error will decrease if we can use activation per-channel quantization. However, channel-wise activation quantization currently could not be performed because it can not map to hardware-accelerate GEMM kernels well.

SmoothQuant is an alternative method of per-channel activation quantization. It divides the input activation by a per-channel smoothing factor $s\in\mathbb R^{C_i} $ , where $C_i$ is the input channel. It also scales the weights accordingly to keep the mathematical equivalence.
$$
Y = (Xdiag(s)^{-1})\cdot(diag(s)W) = \hat{X}\hat{W}
$$
This formula migrates the quantization difficulty from activations to weights. In order to control how much difficulty is shifted to weights, a hyper-parameter named migration strength $\alpha$ is used. 
$$
s_j = max(|X_j|)^\alpha / max(|W_j|)^{1-\alpha}
$$
$j = 1, 2, ...s, C_i$ where j correspond to j-th input channel.

![](./imgs/smoothquant.png)

For most of models such as OPT and BLOOM, $\alpha = 0.5$ is a well-balanced value to split the difficulty of weight and activation quantization. A larger $\alpha$ value could be used on models with more significant activation outliers to migrate more quantization difficulty to weights.

### Our enhancement: 
#### Layer-wise Auto-tuning of $\alpha$.(Yintong)
Instead of using a fixed-value $\alpha$ to control how to split the quantization difficulty, we proposed a method to enable layer-wise auto-tuning of $\alpha$ values. A Layer-wise alpha value is calculated based on a user-defined $\alpha$-value range and then used for smoothing transformation of this layer. Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value of an input LayerNorm op of a transformer block. In our experiments, an $\alpha$ range of [0.3, 0.7] with a step_size of 0.05 is found to be well-balanced one for the majority of models.
#### automatic/more patterns(wenhua)
## Results

| Model\Accuracy        | FP32   | INT8 (w/o SmoothQuant) | INT8 (w/ SmoothQuant) |
| --------------------- | ------ | ---------------------- | --------------------- |
| bigscience/bloom-560m | 65.16% | 64.96%                 | 66.52% (alpha=0.5)    |
| bigscience/bloom-1b7  | 71.55% | 67.61%                 | 72.81% (alpha=0.5)    |
| bigscience/bloom-3b   | 74.06% | 70.73%                 | 74.41% (alpha=0.5)    |
| bigscience/bloom-7b1  | 77.59% | 76.28%                 | 77.18% (alpha=0.5)    |
| bigscience/bloom-176b | 84.17% | 82.13%                 | 83.52% (alpha=0.6)    |
| facebook/opt-125m     | 63.89% | 63.54%                 | 63.91% (alpha=0.5)    |
| facebook/opt-2.7b     | 77.90% | 78.99%                 | 78.91% (alpha=0.5)    |
| facebook/opt-6.7b     | 81.51% | 79.44%                 | 81.58% (alpha=0.5)    |
| EleutherAI/gpt-j-6B   | 79.17% | 78.76%                 | 79.13% (alpha=0.5)    |

# Smooth Quant

1. [Introduction](#Introduction)
2. [Quantization Fundamentals](#Quantization-Fundamentals)
3. [SmoothQuant and Our Enhancement](#SmoothQuant-and-Our-Enhancement)
4. [Validated Models](#Validated-Models)
5. [Usage](#Usage)
6. [Supported Framework Matrix](#Supported-Framework-Matrix)



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


However, due to the model accuracy and computational consumption, per-tensor or per-channel are usually adopted. **In the following part, We will show that per-channel could bring lower quantization loss but has some limitations, that is why normally we use per-channel for weight quantization and per-tensor for activation/input quantization**

#### Per-tensor example

Suppose the weight tensor is：

```python
W = np.array(
    [
        [0.6839, 0.4741, 0.7451],
        [0.9301, 0.1742, 0.6835],
    ]
)
```

According to the formula (1), we need scale $S$ and zero point $Z$ to calculate the integer matrix.

$$
S = \frac{X_{max} - X{min}}{2^b -1} \tag{2}
$$

$$
Z = -round(X_{min/}/S) \tag{3}
$$

The per-tensor quantization function is:

```python
def quantize(x, num_bits=8):
    q_min, q_max = 0, 2.0**num_bits - 1.0
    scale = (np.max(x) - np.min(x)) / (2**num_bits - 1)
    scale = np.clip(scale, 1e-5, None)
    zp = (0 - (np.min(x)) / scale).round()
    q_x = x / scale + zp
    q_x = np.clip(q_x, q_min, q_max).round()
    print(f"scale = {scale}, zp = {zp}")
    return q_x, scale, zp
```

Then we can get the quantized $W_{q}$

```bash
>>> W_q, scale, zp = quantize(W)
scale = 0.0029643137254901962, zp = -59.0
>>> W_q
array([[172., 101., 192.],
       [255.,   0., 172.]])
```

With the value of scale and zp, we can dequantize the tensor.

```python
def dequantize(q_x, scale, zp):
    return scale * (q_x - zp)
```

```bash
>>> W_dq = dequantize(W_q, 0.001, -50)
>>> W_dq
array([[0.222, 0.151, 0.242],
       [0.305, 0.05 , 0.222]])
>>> loss = ((W_dq - W)**2).mean()
>>> loss
0.19833545500000002

>>> W_dq = dequantize(W_q, scale, zp)
>>> W_dq
array([[0.68475647, 0.4742902 , 0.74404275],
       [0.93079451, 0.17489451, 0.68475647]])
>>> loss = ((W_dq - W)**2).mean()
>>> loss
7.384850698449426e-07
```

The difference between $W$ and $W_{dq}$ shows that quantization affects precision and appropriate values of scale and zero point will reduce the loss of precision. 

#### Per-channel example

Similarly, the example of per-channel quantization is as follows:

```python
def quantize_per_channel(x, num_bits=8):
    q_min, q_max = 0, 2.0**num_bits - 1.0
    x_tmp = np.reshape(x, (x.shape[0], -1))
    scales = np.max(x_tmp, axis=-1, keepdims=True) / (2**num_bits - 1)
    zp = (0 - np.min(x_tmp, axis=-1, keepdims=True) / scales).round()
    q_x = x_tmp / scales + zp
    q_x = np.clip(q_x, q_min, q_max).round()
    print(f"scales = {scales}, \n zp = {zp}")
    return q_x, scales, zp


def dequantize_per_channel(q_x, scales, zp):
    print(q_x, scales, zp)
    print(scales * (q_x - zp))
    return scales * (q_x - zp)
```

```bash
>>>W_q, scales, zp = quantize_per_channel(W)
scales = [[0.00292196]
 [0.00364745]],
 zp = [[-162.]
 [ -48.]]
>>>W_q
array([[ 72.,   0.,  93.],
       [207.,   0., 139.]])

>>>W_dq = dequantize_per_channel(W_q, scales, zp)
>>>W_dq
[[0.68373882 0.47335765 0.7451    ]
 [0.9301     0.17507765 0.68207333]]
```

And the loss is

```bash
>>> loss = ((W_dq - W)**2).mean()
>>> loss.item()
5.637846469306487e-07
```

Through this example, we can see that per-channel quantization has finer granularity and has lower loss (loss 5.6378e-07 for per-channel quantization and 7.3849e-07 for per-tensor quantization).

#### Matmul quantization example

For a MatMul in most model, $Y=X \cdot W$, we can quantize both the weights and activations in order to reduce the storage and accelerate inference.
Using per-tensor scale quantization to show the process.

```python
def quantize_per_tensor_absmax(x, n_bits=8):
    scales = np.max(np.abs(x))
    q_max = 2 ** (n_bits - 1) - 1
    scales = np.clip(scales, 1e-5, None) / q_max
    q_x = x / scales
    q_x = np.clip(q_x, -q_max, q_max).round()
    return q_x, scales


def dequantize(q_x, scale):
    return scale * q_x
```

Randomly initialize the $W$ and $Y$, then calculate the result of $Y=X \cdot W$

```bash
>>>W = np.random.randn(2, 3).astype(np.float32)
>>>X = np.random.randn(3, 4).astype(np.float32)
>>>W
array([[-0.75903535, -1.7662522 ,  1.0559074 ],
       [ 0.47551736,  0.33230257,  0.63447773]], dtype=float32)
>>>X
array([[-0.9628984 ,  0.5076066 , -0.54988813,  1.2411681 ],
       [-1.6626304 ,  0.2284153 ,  0.4905207 , -0.11352996],
       [ 1.3270313 , -0.78117365, -0.3452512 ,  0.826362  ]],
      dtype=float32)
>>>Y = np.matmul(W, X)
>>>Y
array([[ 5.068721  , -1.6135774 , -0.81355196,  0.13099377],
       [-0.16839942, -0.17835854, -0.31753427,  1.076779  ]],
      dtype=float32)
```

Quantize weight and activation, matmul(quantize(X), quantize(Y))

```bash
>>>W_q, W_scale = quantize_per_tensor_absmax(W)
>>>X_q, X_scale = quantize_per_tensor_absmax(X)
>>>print(f'{W_q}\n{W_scale}')
>>>print(f'{X_q}\n{X_scale}')
[[ -55. -127.   76.]
 [  34.   24.   46.]]
0.013907497323404147
[[ -74.   39.  -42.   95.]
 [-127.   17.   37.   -9.]
 [ 101.  -60.  -26.   63.]]
0.013091578258304145

>>>Y_q = np.matmul(W_q, X_q)
>>>Y_q
array([[27875., -8864., -4365.,   706.],
       [ -918., -1026., -1736.,  5912.]], dtype=float32)
>>>Y_dq = dequantize(Y_q, W_scale * X_scale)
>>>Y_dq
array([[ 5.0752316 , -1.6138781 , -0.7947403 ,  0.12854218],
       [-0.16714126, -0.18680494, -0.3160754 ,  1.0764043 ]],
      dtype=float32)
```

#### Per-channel limitation

Though per-channel quantization could bring lower quantization error, we could not apply it for activations due to the difficulty of the dequantization. We would prove it in the following image and the zero point of quantization would be ignored for simplicity.

The image on the left presents a normal MatMul forward  with 1x2 input $x$ and 2x2 weight $w$. The results $y$ could be easily obtained by simple mathematics. In the middle image, we apply per-tensor quantization for activations and per-channel quantization for weights; the results after quantization that are denoted by $y_1$ and $y_2$, could be easily dequantized to the float results $y_{fp1}$ and $y_{fp2}$ by per channel scale $1.0/s_1s_x$ and $1.0/s_2s_x$. However, after applying per-channel quantization for activation (right image), we could not dequantize the  $y_1$ and  $y_2$ to float results.

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


Please note that this conversion will make the quantization of weights more difficult, because the scales attached to weights shown above are per-input-channel, while quantization of weights is per-output-channel or per-tensor.

So **the second question is how much difficulty to be migrated**, that is how to choose the **conversion per-channel scale** $s_{x1}$ and $s_{x2}$ from the above image. Different works adopt different ways.

*SPIQ* just adopts the quantization scale of activations as the conversion per-channel scale.

*Outlier suppression* adopts the scale of the preceding layernorm as the conversion per-channel scale.

*Smoothquant* introduces a hyperparameter $\alpha$ as a smooth factor to calculate the conversion per-channel scale and balance the quantization difficulty of activation and weight.

$$
s_j = max(|X_j|)^\alpha/max(|W_j|)^{1-\alpha} \tag{4}
$$

j is the index of the input channels.



<div align="center">
    <img src="./imgs/smoothquant.png" height="250"/>
</div>



For most of the models such as OPT and BLOOM, $\alpha = 0.5$ is a well-balanced value to split the difficulty of weight and activation quantization. A larger $\alpha$ value could be used on models with more significant activation outliers to migrate more quantization difficulty to weights.


### Our enhancement: 

#### Algorithm: Auto-tuning of $\alpha$.

SmoothQuant method aims to split the quantization difficulty of weight and activation by using a fixed-value $\alpha$ for an entire model. However, as the distributions of activation outliers vary not only across different models but also across different layers within a model, we hereby propose a method to obtain operator-wise optimal $\alpha$ values with the ability to tune automatically.

Our proposed method consists of 8 major steps:

-    Hook input minimum and maximum values of operators to be smoothed.
-    Find a list of operators on which smoothquant could be performed.
-    Generate a list of $\alpha$ values of a user-defined range.
-    Calculate smoothing factor using $\alpha$ value, adjust parameters accordingly and forward the adjusted model given an input sample.
-    Perform per-channel quantization_dequantization of weights and per-tensor quantization_dequantization of activations to predict output.
-    Calculate the loss with respect to FP32 output, iterate the previous two steps given each $\alpha$ value and save the loss per alpha.
-    Apply criterion on input operator and obtain the optimal alpha values of a single input sample.
-    Iterate the previous three steps over a number of input samples and save the optimal $\alpha$ values.



Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value. Both alpha range and criterion could be configured in AutoAlphaArgs.

In our experiments, an $\alpha$ range of [0.0, 1.0] with a step_size of 0.1 is found to be well-balanced one for the majority of models.

#### Engineering 

*fully automated*: users only need to pass a model and dataloader.

```python
from neural_compressor_ort.algorithms import Smoother
smoother = Smoother(
        model,
        calibration_data_reader,
        providers=["CPUExecutionProvider"],
    )
smoothed_model = smoother.transform(alpha=0.7) #alpha could 'auto' to enable auto-tuning
```

*support lots of fusing patterns*: when applying the conversion per-channel scales, a mul layer needs to be inserted, which will introduce some overhead. The official code fuses this op to the previous layernorm, while we support more operator types like MatMul, Conv. Currently we only handle the operator whose scale could be fused, we are trying to support other operators, please stay tuned.

## Usage

There are two ways to apply smooth quantization: 1) using a fixed `alpha` for the entire model or 2) determining the `alpha` through auto-tuning.

### Using a fixed `alpha`
To set a fixed alpha for the entire model, users can follow this example:

```python
from neural_compressor_ort.quantization import StaticQuantConfig
config = StaticQuantConfig(
    data_reader,
    extra_options={
        "SmoothQuant": True,
        "SmoothQuantAlpha": 0.5,
        "SmoothQuantFolding": True
    }
)
```
Supported parameters description:

"SmoothQuantAlpha": a float value. Default is 0.5.

"SmoothQuantFolding": whether to fold mul into the previous operator if possible, where mul is required to update the input distribution during smoothing.


### Determining the `alpha` through auto-tuning
Users can search for the best `alpha` at two levels: 1) for the entire model, and 2) for each operator.

#### Auto-tune the `alpha` for the entire model
The tuning process looks for the optimal `alpha` value from a list of `alpha` values provided by the user.
> Please note that, it may a considerable amount of time as the tuning process applies each `alpha` to the entire model and uses the evaluation result on the entire dataset as the metric to determine the best `alpha`.
Here is an example:

```python
from neural_compressor_ort.common.base_tuning import TuningConfig
from neural_compressor_ort.quantization import SmoothQuantConfig, autotune
config = TuningConfig(
    config_set=[SmoothQuantConfig(alpha=np.arange(0.1, 0.5, 0.05).tolist())])
best_model = autotune(
    model_input=model,
    tune_config=config,
    eval_fn=eval_fn,
    calibration_data_reader=data_reader,
)
```
#### Auto-tune the `alpha` for each operator
In this case, the tuning process searches the optimal `alpha` of each operator by evaluating the loss with respect to FP32 output on a few batches of data.
Here is an example:

```python
from neural_compressor_ort.quantization import StaticQuantConfig, quantize
config = StaticQuantConfig(
    data_reader,
    extra_options={
        "SmoothQuant": True,
        "SmoothQuantAlpha": "auto",
        "SmoothQuantCalibIter": 1,
        "AutoAlphaArgs": {
            "alpha_min": 0.3, # min value of auto-tuning alpha search space
            "alpha_max": 0.7, # max value of auto-tuning alpha search space
            "alpha_step": 0.05, # step_size of auto-tuning alpha search space
            "attn_method": "min"
        }
    },
)
quantize(model, output_model_path, config)
```

## Reference

[^1]: Jason, Wei, et al. "Emergent Abilities of Large Language Models". Published in Transactions on Machine Learning Research (2022).

[^2]: Yvinec, Edouard, et al. "SPIQ: Data-Free Per-Channel Static Input Quantization." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

[^3]: Wei, Xiuying, et al. "Outlier suppression: Pushing the limit of low-bit transformer language models." arXiv preprint arXiv:2209.13325 (2022).

[^4]: Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." arXiv preprint arXiv:2211.10438 (2022).

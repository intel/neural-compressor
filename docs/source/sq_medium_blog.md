# SmoothQunat

## INC introduction

Intel® Neural Compressor(INC) is an  open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime, and MXNet). INC aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/), and [MXNet](https://mxnet.apache.org/), as well as Intel extensions such as [Intel Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch). In addition, the tool showcases the key features, typical examples, and broad collaborations as below:

- Support a wide range of Intel hardware such as [Intel Xeon Scalable processor](https://www.intel.com/content/www/us/en/products/details/processors/xeon/scalable.html), [Intel Xeon CPU Max Series](https://www.intel.com/content/www/us/en/products/details/processors/xeon/max-series.html), [Intel Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html), and [Intel Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) with extensive testing; support AMD CPU, ARM CPU, and NVidia GPU through ONNX Runtime with limited testing
- Validate more than 10,000 models such as [Stable Diffusion](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/text-to-image/quantization), [GPT-J](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx), [BERT-Large](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx), and [ResNet50](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx) from popular model hubs such as [Hugging Face](https://huggingface.co/), [Torch Vision](https://pytorch.org/vision/stable/index.html), and [ONNX Model Zoo](https://github.com/onnx/models#models), by leveraging zero-code optimization solution [Neural Coder](https://github.com/intel/neural-compressor/blob/master/neural_coder#what-do-we-offer) and automatic [accuracy-driven](https://github.com/intel/neural-compressor/blob/master/docs/source/design.md#workflow) quantization strategies
- Collaborate with cloud marketplace such as [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/bitnami-launchpad/inc-tensorflow-intel?project=verdant-sensor-286207), [Amazon Web Services](https://aws.amazon.com/marketplace/pp/prodview-yjyh2xmggbmga#pdp-support), and [Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/bitnami.inc-tensorflow-intel), software platforms such as [Alibaba Cloud](https://www.intel.com/content/www/us/en/developer/articles/technical/quantize-ai-by-oneapi-analytics-on-alibaba-cloud.html) and [Tencent TACO](https://new.qq.com/rain/a/20221202A00B9S00), and open AI ecosystem such as [Hugging Face](https://huggingface.co/blog/intel), [PyTorch](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html), [ONNX](https://github.com/onnx/models#models), and [Lightning AI](https://github.com/Lightning-AI/lightning/blob/master/docs/source-pytorch/advanced/post_training_quantization.rst)

**Visit the Intel® Neural Compressor online document website at: https://intel.github.io/neural-compressor.**

## LLM introduction

A Large language mode (LLM) is a language model with billions of weights or more, trained on massive data to solve natural language processing (NLP) and natural language generation (NLG) tasks. Base on large amount of training text such as wikipedia and corpora, LLMs can knowledge about the structure of sentence, the relationship between words and the meaning of whole documents. More complex network structures and more parameters provide LLMs ability to face the complexity and polysemy of natural language.

LLMs are used in a wide variety of applications. They can be used to generate text, such as chatbots and virtual assistants, or fine tuned with a task-specific training for application to downstream tasks, like machine translation, emotion analysis, text classification, fraud detection and etc. 

## LLM deployment challenges
LLMs show excellent performance in many tasks, and research shows that LLMs with bigger number of parmaters can have better performance.
![](./imgs/model_scale_accuracy.png)
Therefore, the scale of LLMs grows exponentially. For example, GPT-2, released in 2019, has 1.5 billion parameters and the number of parameters increase to 175 billions when GPT-3 released in 2020.

Billions or more paramaters make LLMs perform well in various tasks, howerever, also make it more difficult to deploy. Models are usually loaded on servers which have limited memory for infering tasks. The large scale of LLM make the process of inference very slow and even worse, it cannot work if the infrastructure does not meet the requirement.
## Quantization Fundamentals

Quantization is a common compression operation to reduce memory and accelerate inference, therefore, the difficulty of LLM deployment can be alleviate. Quantization convert the floating point matrix to an integer matrix.  `Affine quantization` and `Scale quantization`, also called `asymmetric quantization` and `symmetric quantization`, are two common range mapping techniques used in tensor conversion between different data types.

## SmoothQuant and our enhancement

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

### Our enhancement: Layer-wise Auto-tuning of $\alpha$.
Instead of using a fixed-value $\alpha$ to control how to split the quantization difficulty, we proposed a method to enable layer-wise auto-tuning of $\alpha$ values. A Layer-wise alpha value is calculated based on a user-defined $\alpha$-value range and then used for smoothing transformation of this layer. Multiple criteria (e.g min, max and mean) are supported to determine the $\alpha$ value of an input LayerNorm op of a transformer block. In our experiments, an $\alpha$ range of [0.3, 0.7] with a step_size of 0.05 is found to be well-balanced one for the majority of models.

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
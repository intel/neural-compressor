NVFP4 Quantization
==================

1. [Introduction](#introduction)
2. [Get Started with NVFP4 Quantization API](#get-started-with-nvfp4-quantization-api)
3. [Reference](#reference)

## Introduction

Large language models (LLMs) have revolutionized fields such as natural language understanding, generation, and multimodal processing. As these models grow, their computational and memory requirements increase, making efficient deployment challenging. To address these issues, quantization methods are employed to reduce model size and accelerate inference with minimal loss in accuracy.

NVFP4 is a specialized 4-bit floating-point format (FP4) developed by NVIDIA for deep learning workloads. Compared to traditional INT8 or FP16 formats, NVFP4 offers further reductions in memory footprint and computational resource use, enabling efficient inference for LLMs and other neural networks on supported hardware.

The following table summarizes the NVFP4 quantization format:

<table>
  <tr>
    <th>Format Name</th>
    <th>Element Data type</th>
    <th>Element Bits</th>
    <th>Scaling Block Size</th>
    <th>Scale Data Type</th> 
    <th>Scale Bits</th>
    <th>Global Tensor-Wise Scale Data Type</th> 
    <th>Global Tensor-Wise Scale Bits</th>
  </tr>
  <tr>
    <td>NVFP4</td>
    <td>E2M1</td>
    <td>4</td>
    <td>16</td>
    <td>UE4M3</td> 
    <td>8</td>
    <td>FP32</td> 
    <td>32</td>
  </tr>
</table>

> Note: UE4M3 is the same data type as normal FP8 E4M3, here UE4M3 is named to remind that the sign bit remains 0 and scale is always positive.

### Understanding the Scaling Mechanism

NVFP4 uses a two-level scaling approach to maintain accuracy while reducing precision:

- **Block-wise Scale**: The quantized tensor is divided into blocks of size 16 (the Scaling Block Size). Each block has its own scale factor stored in UE4M3 format (8 bits), which is used to convert the 4-bit E2M1 quantized values back to a higher precision representation. This fine-grained scaling helps preserve local variations in the data.

- **Global Tensor-Wise Scale**: In addition to the block-wise scales, a single FP32 (32-bit) scale factor is applied to the entire tensor. This global scale provides an additional level of normalization for the whole weight or activation tensor. For activations, this global scale is static (computed during calibration and fixed during inference) to optimize performance.

The dequantization formula can be expressed as:

`dequantized_value = quantized_value × block_scale × global_scale`

This hierarchical scaling strategy balances compression efficiency with numerical accuracy, enabling NVFP4 to maintain model performance while significantly reducing memory footprint.

At similar accuracy levels, NVFP4 can deliver lower memory usage and improved compute efficiency for multiply-accumulate operations compared to higher-precision formats. Neural Compressor supports post-training quantization to NVFP4, providing recipes and APIs for users to quantize LLMs easily.

## Get Started with NVFP4 Quantization API

To quantize a model to the NVFP4 format, use the AutoRound Quantization API as shown below.

```python
from neural_compressor.torch.quantization import AutoRoundConfig, prepare, convert
from transformers import AutoModelForCausalLM, AutoTokenizer

fp32_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
output_dir = "./saved_inc"

# quantization configuration
# `iters=0` means RTN (fast, no optimization); use default `iters=200` if accuracy is poor
quant_config = AutoRoundConfig(
    tokenizer=tokenizer,  # Tokenizer for processing calibration data
    scheme="NVFP4",  # NVFP4 quantization scheme
    iters=0,  # Number of optimization iterations (default: 200)
    export_format="llm_compressor",  # Export format for the quantized model
    output_dir=output_dir,  # Directory to save the quantized model (default: "temp_auto_round")
)

# quantize the model and save to output_dir
model = prepare(model=fp32_model, quant_config=quant_config)
model = convert(model)

# loading
model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype="auto", device_map="auto")

# inference
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
```

## Reference

[1]: NVIDIA, Introducing NVFP4 for efficient and accurate low-precision inference,NVIDIA Developer Blog, Jun. 2025. [Online]. Available: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

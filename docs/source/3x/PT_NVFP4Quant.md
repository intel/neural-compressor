NVFP4 Quantization
==================

1. [Introduction](#introduction)
2. [Get Started with NVFP4 Quantization API](#get-started-with-nvfp4-quantization-api)
3. [Examples](#examples)
4. [Reference](#reference)

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
    <th>Global Scale Data Type</th> 
    <th>Global Scale Bits</th>
  </tr>
  <tr>
    <td>NVFP4</td>
    <td>E2M1</td>
    <td>4</td>
    <td>16</td>
    <td>E4M3</td> 
    <td>8</td>
    <td>FP32</td> 
    <td>32</td>
  </tr>
</table>

At similar accuracy levels, NVFP4 can deliver lower memory usage and improved compute efficiency for multiply-accumulate operations compared to higher-precision formats. Neural Compressor supports post-training quantization to NVFP4, providing recipes and APIs for users to quantize LLMs easily. To provide the best performance, the global scale for activation is static.

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
quant_config = AutoRoundConfig(
    tokenizer=tokenizer,
    nsamples=32,
    seqlen=32,
    iters=20,
    scheme="NVFP4",  # NVFP4 format
    export_format="llm_compressor",
    output_dir=output_dir,  # default is "temp_auto_round"
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
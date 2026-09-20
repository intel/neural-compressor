# Examples

Intel® Neural Compressor validated examples with multiple compression techniques, including quantization, pruning, knowledge distillation and orchestration.

# PyTorch Examples

## Weight-Activation Quantization
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Method </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
<tr>
    <td>deepseek-ai/DeepSeek-V4</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP8/MXFP4)</td>
    <td><a href="./pytorch/llm/deepseekv4">link</a></td>
</tr>
<tr>
    <td>deepseek-ai/DeepSeek-R1</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP8/MXFP4/NVFP4)</td>
    <td><a href="./pytorch/llm/deepseek">link</a></td>
</tr>
<tr>
    <td>Qwen/Qwen3-235B-A22B</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP8/MXFP4)</td>
    <td><a href="./pytorch/llm/qwen">link</a></td>
</tr>
<tr>
    <td>moonshotai/Kimi-K2.6</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP4)</td>
    <td><a href="./pytorch/llm/kimi-glm">link</a></td>
</tr>
<tr>
    <td>zai-org/GLM-5.2</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP4)</td>
    <td><a href="./pytorch/llm/kimi-glm">link</a></td>
</tr>
<tr>
    <td>MiniMaxAI/MiniMax-M2.7</td>
    <td>Natural Language Processing</td>
    <td>Quantization (MXFP4)</td>
    <td><a href="./pytorch/llm/minimax">link</a></td>
</tr>
<tr>
    <td>Framepack</td>
    <td>Image + Text to Video</td>
    <td>Quantization (MXFP8/FP8)</td>
    <td><a href="./pytorch/diffusion_model/framepack">link</a></td>
</tr>
<tr>
    <td>SDXL</td>
    <td>Text to Image</td>
    <td>Quantization (MXFP8)</td>
    <td><a href="./pytorch/diffusion_model/sdxl">link</a></td>
</tr>
<tr>
    <td>FLUX.1-dev</td>
    <td>Text to Image</td>
    <td>Quantization (MXFP8/FP8)</td>
    <td><a href="./pytorch/diffusion_model/flux">link</a></td>
</tr>
<tr>
    <td>Wan-AI/Wan2.2-I2V-A14B-Diffusers</td>
    <td>Image to Video</td>
    <td>Quantization (MXFP8/FP8)</td>
    <td><a href="./pytorch/diffusion_model/wan">link</a></td>
</tr>
<tr>
    <td>Wan-AI/Wan2.2-T2V-A14B-Diffusers</td>
    <td>Text to Video</td>
    <td>Quantization (MXFP8/FP8)</td>
    <td><a href="./pytorch/diffusion_model/wan">link</a></td>
</tr>
<tr>
    <td>Wan-AI/Wan2.2-S2V-14B</td>
    <td>Subject to Video</td>
    <td>Quantization (MXFP8/FP8)</td>
    <td><a href="./pytorch/diffusion_model/wan">link</a></td>
</tr>
<tr>
    <td rowspan="2">Llama-3.1-8B-Instruct</td>
    <td rowspan="2">Natural Language Processing</td>
    <td>Mixed Precision (MXFP4+MXFP8)</td>
    <td><a href="./pytorch/llm/llama3/README.html#llama-31-8b-mxfp4-mixed-with-mxfp8-target_bits78">link</a></td>
</tr>
<tr>
    <td>Quantization (MXFP4/MXFP8/NVFP4)</td>
    <td><a href="./pytorch/llm/llama3/README.html#demo-mxfp4-mxfp8-nvfp4-unvfp4">link</a></td>
</tr>
<tr>
    <td rowspan="2">Llama-3.1-70B-Instruct</td>
    <td rowspan="2">Natural Language Processing</td>
<tr>
    <td>Quantization (MXFP8/NVFP4/uNVFP4)</td>
    <td><a href="./pytorch/llm/llama3/README.html#llama-31-70b-mxfp8">link</a></td>
</tr>
<tr>
    <td rowspan="2">Llama-3.3-70B-Instruct</td>
    <td rowspan="2">Natural Language Processing</td>
    <td>Mixed Precision (MXFP4+MXFP8)</td>
    <td><a href="./pytorch/llm/llama3/README.html#llama-33-70b-mxfp4-mixed-with-mxfp8-target_bits58">link</a></td>
</tr>
<tr>
    <td>Quantization (MXFP4/MXFP8/NVFP4)</td>
    <td><a href="./pytorch/llm/llama3/README.html#demo-mxfp4-mxfp8-nvfp4-unvfp4">link</a></td>
</tr>
</tbody>
</table>

## Weight-only Quantization
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Method </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
<tr>
    <td>gpt_j</td>
    <td>Natural Language Processing</td>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/llm/others/weight_only">link</a></td>
</tr>
<tr>
    <td>llama2_7b</td>
    <td>Natural Language Processing</td>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/llm/others/weight_only">link</a></td>
</tr>
<tr>
    <td>opt_125m</td>
    <td>Natural Language Processing</td>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/llm/others/weight_only">link</a></td>
</tr>
</tbody>
</table>

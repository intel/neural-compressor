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

# TensorFlow Examples (Deprecated)

## Quantization

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Method</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
 <tr>
    <td>bert_large_squad_model_zoo</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/bert_large_squad_model_zoo/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>transformer_lt</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/transformer_lt/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>inception_v3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/inception_v3/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>mobilenetv2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/mobilenet_v2/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>resnetv2_50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/resnet_v2_50/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>vgg16</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/vgg16/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>ViT</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/vision_transformer/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>GraphSage</td>
    <td>Graph Networks</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/graph_networks/graphsage/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>yolo_v5</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/yolo_v5/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>faster_rcnn_resnet50</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/faster_rcnn_resnet50/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>mask_rcnn_inception_v2</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/mask_rcnn_inception_v2/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>ssd_mobilenet_v1</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/ssd_mobilenet_v1/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>wide_deep_large_ds</td>
    <td>Recommendation</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/recommendation/wide_deep_large_ds/quantization/ptq">link</a></td>
</tr>
<tr>
    <td>3dunet-mlperf</td>
    <td>Semantic Image Segmentation</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/semantic_image_segmentation/3dunet-mlperf/quantization/ptq">link</a></td>
</tr>

</tbody>
</table>


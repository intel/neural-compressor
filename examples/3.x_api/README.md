# Examples

Intel® Neural Compressor validated examples with multiple compression techniques, including quantization, pruning, knowledge distillation and orchestration. Part of the validated cases can be found in the example tables, and the release data is available [here](../docs/source/validated_model_list.md).


# PyTorch Examples

## Quantization
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
    <td rowspan="2">gpt_j</td>
    <td rowspan="2">Natural Language Processing</td>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/weight_only">link</a></td>
</tr>
<tr>
    <td>Static Quantization (IPEX)</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/ipex">link</a></td>
</tr>
<tr>
    <td rowspan="2">llama2_7b</td>
    <td rowspan="2">Natural Language Processing</td>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/weight_only">link</a></td>
</tr>
<tr>
    <td>Static Quantization (IPEX)</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/ipex">link</a></td>
</tr>
<tr>
    <td rowspan="3">opt_125m</td>
    <td rowspan="3">Natural Language Processing</td>
    <td>Static Quantization (IPEX)</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/ipex">link</a></td>
</tr>
<tr>
    <td>Static Quantization (PT2E)</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/static_quant/pt2e">link</a></td>
</tr>
<tr>
    <td>Weight-Only Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/weight_only">link</a></td>
</tr>
<tr>
    <td rowspan="2">resnet18</td>
    <td rowspan="2">Image Recognition</td>
    <td>Mixed Precision</td>
    <td><a href="./pytorch/cv/mixed_precision">link</a></td>
</tr>
<tr>
    <td>Static Quantization</td>
    <td><a href="./pytorch/cv/static_quant">link</a></td>
</tr>
</tbody>
</table>


# TensorFlow Examples

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


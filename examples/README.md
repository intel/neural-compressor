Examples
==========
Intel® Neural Compressor validated examples with multiple compression techniques, including quantization, pruning, knowledge distillation and orchestration. Part of the validated cases can be found in the example tables, and the release data is available [here](../docs/source/validated_model_list.md).
> Note: The example marked with `*` means it still use 1.x API.

# Quick Get Started Notebook Examples
* [Quick Get Started Notebook of Intel® Neural Compressor for ONNXRuntime](/examples/notebook/onnxruntime/Quick_Started_Notebook_of_INC_for_ONNXRuntime.ipynb)

# Helloworld Examples

* [tf_example1](/examples/helloworld/tf_example1): quantize with built-in dataloader and metric.
* [tf_example2](/examples/helloworld/tf_example2): quantize keras model with customized metric and dataloader.
* [tf_example3](/examples/helloworld/tf_example3): convert model with mix precision.
* [tf_example4](/examples/helloworld/tf_example4): quantize checkpoint with dummy dataloader.
* [tf_example5](/examples/helloworld/tf_example5): config performance and accuracy measurement.
* [tf_example6](/examples/helloworld/tf_example6): use default user-facing APIs to quantize a pb model.
* [tf_example7](/examples/helloworld/tf_example7): quantize and benchmark with pure python API.

# TensorFlow Examples
## Quantization
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50 V1.0</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet50_v1/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet50_v1_5/quantization/ptq">pb</a> / <a href="./keras/image_recognition/resnet50/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet101/quantization/ptq">pb</a> / <a href="./keras/image_recognition/resnet101/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>MobileNet V1</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/mobilenet_v1/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/mobilenet_v2/quantization/ptq">pb</a> / <a href="./keras/image_recognition/mobilenet_v2/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>MobileNet V3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/mobilenet_v3/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Inception V1</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_v1/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Inception V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_v2/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_v3/quantization/ptq">pb</a> / <a href="./keras/image_recognition/inception_v3/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>Inception V4</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_v4/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Inception ResNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_resnet_v2/quantization/ptq">pb</a> / <a href="./keras/image_recognition/inception_resnet_v2/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vgg16/quantization/ptq">pb</a> / <a href="./keras/image_recognition/vgg16/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vgg19/quantization/ptq">pb</a> / <a href="./keras/image_recognition/vgg19/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2 50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet_v2_50/quantization/ptq">pb</a> / <a href="./keras/image_recognition/resnetv2_50/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2 101</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet_v2_101/quantization/ptq">pb</a> / <a href="./keras/image_recognition/resnetv2_101/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2 152</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet_v2_152/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>DenseNet121</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/densenet121/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>DenseNet161</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/densenet161/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>DenseNet169</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/densenet169/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>EfficientNet B0</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/efficientnet-b0/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>Xception</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./keras/image_recognition/xception/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="./tensorflow/image_recognition/resnet_v2/quantization/qat">keras</a> </td>
  </tr>
  <tr>
    <td>EfficientNet V2 B0</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/SavedModel/efficientnet_v2_b0/quantization/ptq">SavedModel</a></td>
  </tr>
    <td>BERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/bert_base_mrpc/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>BERT large SQuAD (Model Zoo)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/bert_large_squad_model_zoo/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>BERT large SQuAD</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/bert_large_squad/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>DistilBERT base</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/distilbert_base/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Transformer LT</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/transformer_lt/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Transformer LT MLPerf</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/transformer_lt_mlperf/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>SSD ResNet50 V1</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/ssd_resnet50_v1/quantization/ptq">pb</a> / <a href="./tensorflow/object_detection/tensorflow_models/ssd_resnet50_v1/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>SSD MobileNet V1</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/ssd_mobilenet_v1/quantization/ptq">pb</a> / <a href="./tensorflow/object_detection/tensorflow_models/ssd_mobilenet_v1/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>Faster R-CNN Inception ResNet V2</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_inception_resnet_v2/quantization/ptq">pb</a> / <a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_inception_resnet_v2/quantization/ptq">SavedModel</a></td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet101</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_resnet101/quantization/ptq">pb</a> / <a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_resnet101/quantization/ptq">SavedModel</a></td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet50</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_resnet50/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Mask R-CNN Inception V2</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/mask_rcnn_inception_v2/quantization/ptq">pb</a> / <a href="./tensorflow/object_detection/tensorflow_models/mask_rcnn_inception_v2/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>SSD ResNet34</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/ssd_resnet34/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>YOLOv3</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/object_detection/yolo_v3/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>Wide & Deep</td>
    <td>Recommendation</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/recommendation/wide_deep_large_ds/quantization/ptq">pb</a></td>
  </tr> 
  <tr>
    <td>Arbitrary Style Transfer</td>
    <td>Style Transfer</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/style_transfer/arbitrary_style_transfer/quantization/ptq">ckpt</a></td>
  </tr>
  <tr>
    <td>OPT</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/large_language_models/quantization/ptq/smoothquant">pb (smooth quant)</a></td>
  </tr>
  <tr>
    <td>GPT2</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/nlp/large_language_models/quantization/ptq/smoothquant">pb (smooth quant)</a></td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vision_transformer/">pb</a></td>
  </tr>
  <tr>
    <td>GraphSage</td>
    <td>Graph Networks</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/graph_networks/graphsage/">pb</a></td>
  </tr>
</tbody>
</table>

## Distillation
<table>
<thead>
  <tr>
    <th>Student Model</th>
    <th>Teacher Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MobileNet</td>
    <td>DenseNet201</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/distillation">pb</a></td>
  </tr>
</tbody>
</table>

## Pruning
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet V2</td>
    <td>Image Recognition</td>
    <td>Structured (4x1, 2in4)</td>
    <td><a href="./tensorflow/image_recognition/resnet_v2/pruning/magnitude">keras</a></td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>Image Recognition</td>
    <td>Structured (4x1, 2in4)</td>
    <td><a href="./tensorflow/image_recognition/ViT/pruning/magnitude">keras</a></td>
  </tr>
</tbody>
</table>

## Model Export
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50 V1</td>
    <td>Image Recognition</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet50_v1/export">int8 fp32</a></td>
  </tr>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>Image Recognition</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet50_v1_5/export">int8 fp32</a></td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>Image Recognition</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/mobilenet_v2/export">int8 fp32</a></td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>Image Recognition</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vgg16/export">int8 fp32</a></td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet50</td>
    <td>Object Detection</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/faster_rcnn_resnet50/export">int8 fp32</a></td>
  </tr>
  <tr>
    <td>SSD MobileNet V1</td>
    <td>Object Detection</td>
    <td>TF2ONNX</td>
    <td><a href="./tensorflow/object_detection/tensorflow_models/ssd_mobilenet_v1/export">int8 fp32</a></td>
  </tr>
</tbody>
</table>

# PyTorch Examples
## Quantization
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet18</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>ResNet18</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/qat/fx">fx</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/qat/fx">fx</a></td>
  </tr>
  <tr>
    <td>ResNeXt101_32x16d_wsl</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>Se_ResNeXt50_32x4d</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/se_resnext/quantization/ptq/fx">fx</a></td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>PeleeNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/peleenet/quantization/ptq/fx">fx</a></td>
  </tr>
  <tr>
    <td>ResNeSt50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/resnest/quantization/ptq/fx">fx</a></td>
  </tr>
  <tr>
    <td>3D-UNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/3d-unet/quantization/ptq/fx">fx</a></td>
  </tr>
  <tr>
    <td>SSD ResNet34</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/object_detection/ssd_resnet34/quantization/ptq/fx">fx</a> / <a href="./pytorch/object_detection/ssd_resnet34/quantization/ptq/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>YOLOv3</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/object_detection/yolo_v3/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>DLRM</td>
    <td>Recommendation</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/recommendation/dlrm/quantization/ptq/ipex">ipex</a> / <a href="./pytorch/recommendation/dlrm/quantization/ptq/fx">fx</a></td>
  </tr>
  <tr>
    <td>HuBERT</td>
    <td>Speech Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/speech_recognition/torchaudio_models/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>HuBERT</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/speech_recognition/torchaudio_models/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>BlendCNN</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/blendcnn/quantization/ptq/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>bert-large-uncased-whole-word-masking-finetuned-squad</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/fx">fx</a> / <a href="./pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>distilbert-base-uncased-distilled-squad</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>yoshitomo-matsubara/bert-large-uncased-rte</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>Intel/xlm-roberta-base-mrpc</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/distilbert-base-uncased-MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/albert-base-v2-MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>Intel/xlm-roberta-base-mrpc</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>yoshitomo-matsubara/bert-large-uncased-rte</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>Intel/bert-base-uncased-mrpc</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/bert-base-uncased-CoLA</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/bert-base-uncased-STS-B</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>gchhablani/bert-base-cased-finetuned-sst2</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>ModelTC/bert-base-uncased-rte</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/bert-base-uncased-QNLI</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>yoshitomo-matsubara/bert-large-uncased-cola</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/distilbert-base-uncased-MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>Intel/xlnet-base-cased-mrpc</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>textattack/roberta-base-MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>Intel/camembert-base-mrpc</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>t5-small</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>Helsinki-NLP/opus-mt-en-ro</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>lvwerra/pegasus-samsum</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/summarization/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>google/reformer-crime-and-punishment</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6B</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx">fx</a> / <a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/ipex/smooth_quant">smooth quant</a></td>
  </tr>
  <tr>
    <td>EleutherAI/gpt-j-6B</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Weight Only Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_weight_only">weight_only</a></td>
  </tr>
  <tr>
    <td>abeja/gpt-neox-japanese-2.7b</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>bigscience/bloom</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/ipex/smooth_quant">smooth quant</a></td>
  </tr>
  <tr>
    <td>facebook/opt</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/ipex/smooth_quant">smooth quant</a></td>
  </tr>
  <tr>
    <td>SD Diffusion</td>
    <td>Text to Image</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-to-image/quantization">fx</a></td>
  </tr>
  <tr>
    <td>openai/whisper-large</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/speech_recognition/whisper_large/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>
  <tr>
    <td>torchaudio/wav2vec2</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./pytorch/speech_recognition/torchaudio_models/quantization/ptq_dynamic/fx">fx</a></td>
  </tr>

</tbody>
</table>

## Quantization with [Intel® Extension for Transformers](https://github.com/intel/intel-extension-for-transformers) based on Intel® Neural Compressor
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>T5 Large</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/summarization/quantization">fx</a></td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Flan T5 Large</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="https://github.com/intel/intel-extension-for-transformers/tree/main/examples/huggingface/pytorch/summarization/quantization">fx</a></td>
  </tr>
</tbody>
</table>

## Pruning
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Pruning Type </th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>Natural Language Processing (text classification)</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Bert-mini</td>
    <td>Natural Language Processing (text classification)</td>
    <td>Structured (4x1, 2in4, per channel), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>Natural Language Processing (question answering)</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Bert-mini</td>
    <td>Natural Language Processing (question answering)</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Bert-base-uncased</td>
    <td>Natural Language Processing (question answering)</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Bert-large</td>
    <td>Natural Language Processing (question answering)</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Flan-T5-small</td>
    <td>Natural Language Processing (translation)</td>
    <td>Structured (4x1)</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/translation/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>YOLOv5s6</td>
    <td>Object Detection</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/object_detection/yolo_v5/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Structured (2x1)</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/image_recognition/ResNet50/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>Bert-base</td>
    <td>Question Answering</td>
    <td>Structured (channel, multi-head attention)</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/model_slim/">eager</a></td>
  </tr>
  <tr>
    <td>Bert-large</td>
    <td>Question Answering</td>
    <td>Structured (channel, multi-head attention)</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/model_slim/">eager</a></td>
  </tr>
</tbody>
</table>

## Distillation
<table>
<thead>
  <tr>
    <th>Student Model</th>
    <th>Teacher Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CNN-2</td>
    <td>CNN-10</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/image_recognition/CNN-2/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>MobileNet V2-0.35</td>
    <td>WideResNet40-2</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/image_recognition/MobileNetV2-0.35/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet18|ResNet34|ResNet50|ResNet101</td>
    <td>ResNet18|ResNet34|ResNet50|ResNet101</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet18|ResNet34|ResNet50|ResNet101</td>
    <td>ResNet18|ResNet34|ResNet50|ResNet101</td>
    <td>Image Recognition</td>
    <td>Self Distillation</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/self_distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>VGG-8</td>
    <td>VGG-13</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/image_recognition/VGG-8/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>BlendCNN</td>
    <td>BERT-Base</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/blendcnn/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>DistilBERT</td>
    <td>BERT-Base</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>BiLSTM</td>
    <td>RoBERTa-Base</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>TinyBERT</td>
    <td>BERT-Base</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>BERT-3</td>
    <td>BERT-Base</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/distillation/eager">eager</a></td>
  </tr>
  <tr>
    <td>DistilRoBERTa</td>
    <td>RoBERTa-Large</td>
    <td>Natural Language Processing</td>
    <td>Knowledge Distillation</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/distillation/eager">eager</a></td>
  </tr>
</tbody>
</table>

## Orchestration
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Multi-shot: Pruning and PTQ<br></td>
    <td><a href="./pytorch/image_recognition/torchvision_models/optimization_pipeline/prune_and_ptq/fx">link</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>One-shot: QAT during Pruning<br></td>
    <td><a href="./pytorch/image_recognition/torchvision_models/optimization_pipeline/qat_during_prune/fx">link</a></td>
  </tr>
  <tr>
    <td>Intel/bert-base-uncased-sparse-90-unstructured-pruneofa</td>
    <td>Natural Language Processing (question-answering)</td>
    <td>One-shot: Pruning, Distillation and QAT<br></td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/optimization_pipeline/prune_once_for_all/fx">link</a></td>
  </tr>
  <tr>
    <td>Intel/bert-base-uncased-sparse-90-unstructured-pruneofa</td>
    <td>Natural Language Processing (text-classification)</td>
    <td>One-shot: Pruning, Distillation and QAT<br></td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/optimization_pipeline/prune_once_for_all/fx">link</a></td>
  </tr>
</tbody>
</table>

## Model Export
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach</th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet18</td>
    <td>Image Recognition</td>
    <td>PT2ONNX</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/export/fx">int8 fp32</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>PT2ONNX</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/export/fx">int8 fp32</a></td>
  </tr>
  <tr>
    <td>bert base MRPC</td>
    <td>Natural Language Processing</td>
    <td>PT2ONNX</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/export/fx">int8 fp32</a></td>
  </tr>
  <tr>
    <td>bert large MRPC</td>
    <td>Natural Language Processing</td>
    <td>PT2ONNX</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/export/fx">int8 fp32</a></td>
  </tr>
</tbody>
</table>

# ONNX Runtime Examples
## Quantization

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Approach </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/resnet50_torchvision/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 MLPerf</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/resnet50_mlperf/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/resnet50_mlperf/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/vgg16/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/vgg16/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/mobilenet_v2/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/mobilenet_v2/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>MobileNet V3 MLPerf</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/mobilenet_v3/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/mobilenet_v3/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>AlexNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/alexnet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/alexnet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>CaffeNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/caffenet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/caffenet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>DenseNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/densenet/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>EfficientNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/efficientnet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/efficientnet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>FCN (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/fcn/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/fcn/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>GoogleNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/googlenet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/googlenet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Inception V1 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/inception/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/inception/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>MNIST (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/mnist/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>MobileNet V2 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/mobilenet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/mobilenet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/resnet50/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/resnet50/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>ShuffleNet V2 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/shufflenet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/shufflenet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>SqueezeNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/squeezenet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/squeezenet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>VGG16 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/vgg16/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/vgg16/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>ZFNet (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/zfnet/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/zfnet/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>ArcFace (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/arcface/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>CodeBert</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/nlp/huggingface_model/code_detection/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>CodeBert</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/huggingface_model/code_detection/quantization/ptq_dynamic">integerops</a></td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/nlp/bert/quantization/ptq_static">integerops</a> / <a href="./onnxrt/nlp/bert/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/bert/quantization/ptq_dynamic">integerops</a></td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/distilbert/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/distilbert/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Mobile bert MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/mobilebert/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/mobilebert/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Roberta base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/roberta/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/roberta/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>BERT SQuAD (ONNX Model Zoo)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/bert-squad/quantization/ptq_dynamic">integerops</a> </td>
  </tr>
  <tr>
    <td>GPT2 lm head WikiText (ONNX Model Zoo)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/gpt2/quantization/ptq_dynamic">integerops</a></td>
  </tr>
  <tr>
    <td>MobileBERT SQuAD MLPerf (ONNX Model Zoo)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/mobilebert/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/onnx_model_zoo/mobilebert/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>BiDAF (ONNX Model Zoo)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/BiDAF/quantization/ptq_dynamic">integerops</a></td>
  </tr>
  <tr>
    <td>BERT base uncased MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Roberta base MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>XLM Roberta base MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Camembert base MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>MiniLM L12 H384 uncased MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2 (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Albert base v2 SST-2 (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>MiniLM L6 H384 uncased SST-2 (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>BERT base cased MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Electra small discriminator MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>BERT mini MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Xlnet base cased MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> /  <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>BART large MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>DeBERTa v3 base MRPC (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>DistilBert base uncased SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>BERT large uncased whole word masking SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Roberta large SQuAD v2 (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
        <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>GPT2 WikiText (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/language_modeling/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/language_modeling/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>DistilGPT2 WikiText (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/language_modeling/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/language_modeling/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>LayoutLMv3 FUNSD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlmv3/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlmv3/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>LayoutLMv2 FUNSD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlmv2/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlmv2/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>LayoutLM FUNSD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlm/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/token_classification/layoutlm/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>SSD MobileNet V1</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/ssd_mobilenet_v1/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/ssd_mobilenet_v1/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>SSD MobileNet V2</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/ssd_mobilenet_v2/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/ssd_mobilenet_v2/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Table Transformer</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/table_transformer/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/ssd_mobilenet_v1/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/onnx_model_zoo/ssd_mobilenet_v1/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>DUC (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/DUC/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>Faster R-CNN (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/faster_rcnn/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/onnx_model_zoo/faster_rcnn/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Mask R-CNN (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/mask_rcnn/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/onnx_model_zoo/mask_rcnn/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>SSD (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/ssd/quantization/ptq_static">qlinearops</a> / <a href="./onnxrt/object_detection/onnx_model_zoo/ssd/quantization/ptq_static">qdq</a></td>
  </tr>
  <tr>
    <td>Tiny YOLOv3 (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/yolov3/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>YOLOv3 (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/yolov3/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>YOLOv4 (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/yolov4/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>Emotion FERPlus (ONNX Model Zoo)</td>
    <td>Body Analysis</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/emotion_ferplus/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>Ultra Face (ONNX Model Zoo)</td>
    <td>Body Analysis</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/ultraface/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>GPT-J-6B (HuggingFace)</td>
    <td>Text Generation</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/text_generation/gptj/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_generation/gptj/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Llama-7B (HuggingFace)</td>
    <td>Text Generation</td>
    <td>Static Quantization</td>
    <td>
      <a href="./onnxrt/nlp/huggingface_model/text_generation/llama/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
</tbody>
</table>

# Notebook Examples

* *[BERT Mini SST2 performance boost with INC](/examples/notebook/bert_mini_distillation): train a BERT-Mini model on SST-2 dataset through distillation, and leverage quantization to accelerate the inference while maintaining the accuracy using Intel® Neural Compressor.
* [Performance of FP32 Vs. INT8 ResNet50 Model](/examples/notebook/perf_fp32_int8_tf): compare existed FP32 & INT8 ResNet50 model directly.
* [Intel® Neural Compressor Sample for PyTorch*](/examples/notebook/pytorch/alexnet_fashion_mnist): an End-To-End pipeline to build up a CNN model by PyTorch to recognize fashion image and speed up AI model by Intel® Neural Compressor.
* [Intel® Neural Compressor Sample for TensorFlow*](/examples/notebook/tensorflow/alexnet_mnist): an End-To-End pipeline to build up a CNN model by TensorFlow to recognize handwriting number and speed up AI model by Intel® Neural Compressor.
* [Accelerate VGG19 Inference on Intel® Gen4 Xeon® Sapphire Rapids](/examples/notebook/tensorflow/vgg19_ibean): an End-To-End pipeline to train VGG19 model by transfer learning based on pre-trained model from [TensorFlow Hub](https://tfhub.dev); quantize it by Intel® Neural Compressor on Intel® Gen4 Xeon® Sapphire Rapids.


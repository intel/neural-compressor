Examples
==========
Intel® Neural Compressor validated examples with multiple compression techniques, including quantization, pruning, knowledge distillation and orchestration. Part of the validated cases can be found in the example tables, and the release data is available [here](../docs/source/validated_model_list.md).
> Note: 2.0 API migration work in progress, the example marked with `*` means it still using 1.x API.

## Helloworld Examples

* [tf_example1](/examples/helloworld/tf_example1): quantize with built-in dataloader and metric.
* [tf_example2](/examples/helloworld/tf_example2): quantize keras model with customized metric and dataloader.
* [tf_example3](/examples/helloworld/tf_example3): convert model with mix precision.
* [tf_example4](/examples/helloworld/tf_example4): quantize checkpoint with dummy dataloader.
* [tf_example5](/examples/helloworld/tf_example5): config performance and accuracy measurement.
* [tf_example6](/examples/helloworld/tf_example6): use default user-facing APIs to quantize a pb model.
* [tf_example7](/examples/helloworld/tf_example7): quantize and benchmark with pure python API.

## Notebook Examples

* *[BERT Mini SST2 performance boost with INC](/examples/notebook/bert_mini_distillation): train a BERT-Mini model on SST-2 dataset through distillation, and leverage quantization to accelerate the inference while maintaining the accuracy using Intel® Neural Compressor.
* [Performance of FP32 Vs. INT8 ResNet50 Model](/examples/notebook/perf_fp32_int8_tf): compare existed FP32 & INT8 ResNet50 model directly.
* *[Intel® Neural Compressor Sample for PyTorch*](/examples/notebook/pytorch/alexnet_fashion_mnist): an End-To-End pipeline to build up a CNN model by PyTorch to recognize fashion image and speed up AI model by Intel® Neural Compressor.
* *[Intel® Neural Compressor Sample for TensorFlow*](/examples/notebook/tensorflow/alexnet_mnist): an End-To-End pipeline to build up a CNN model by TensorFlow to recognize handwriting number and speed up AI model by Intel® Neural Compressor.
* *[Accelerate VGG19 Inference on Intel® Gen4 Xeon® Sapphire Rapids](/examples/notebook/tensorflow/vgg19_ibean): an End-To-End pipeline to train VGG19 model by transfer learning based on pre-trained model from [TensorFlow Hub](https://tfhub.dev); quantize it by Intel® Neural Compressor on Intel® Gen4 Xeon® Sapphire Rapids.

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
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet50_v1_5/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet101/quantization/ptq">pb</a></td>
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
    <td><a href="./tensorflow/image_recognition/tensorflow_models/mobilenet_v2/quantization/ptq">pb</a> / <a href="./tensorflow/image_recognition/keras_models/mobilenet_v2/quantization/ptq">keras</a></td>
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
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_v3/quantization/ptq">pb</a></td>
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
    <td><a href="./tensorflow/image_recognition/tensorflow_models/inception_resnet_v2/quantization/ptq">pb</a></td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vgg16/quantization/ptq">pb</a> / <a href="./tensorflow/image_recognition/keras_models/vgg16/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/vgg19/quantization/ptq">pb</a> / <a href="./tensorflow/image_recognition/keras_models/vgg19/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2 50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet_v2_50/quantization/ptq">pb</a> / <a href="./tensorflow/image_recognition/keras_models/resnetv2_50/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet V2 101</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/tensorflow_models/resnet_v2_101/quantization/ptq">pb</a> / <a href="./tensorflow/image_recognition/keras_models/resnetv2_101/quantization/ptq">keras</a></td>
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
    <td>MNIST </td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="./tensorflow/image_recognition/keras_models/mnist/quantization/qat">keras</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/resnet50/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet50 Fashion</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/resnet50_fashion/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/resnet101/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/inception_v3/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>Inception Resnet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/inception_resnet_v2/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>Xception</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./tensorflow/image_recognition/keras_models/xception/quantization/ptq">keras</a></td>
  </tr>
  <tr>
    <td>*ResNet V2</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/tensorflow/image_recognition/resnet_v2/quantization/qat">keras</a> </td>
  </tr>
  <tr>
    <td>*EfficientNet V2 B0</td>
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
</tbody>
</table>

## Pruning
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Domain</th>
    <th>Pruning Type </th>
    <th>Approach </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Inception V3</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./tensorflow/image_recognition/inception_v3/pruning/magnitude">pb</a></td>
  </tr>
  <tr>
    <td>ResNet V2</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./tensorflow/image_recognition/resnet_v2/pruning/magnitude">pb</a></td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./tensorflow/image_recognition/ViT/pruning/magnitude">ckpt</a></td>
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
    <th>Approach </th>
    <th>Examples</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>*MobileNet</td>
    <td>DenseNet201</td>
    <td>Image Recognition</td>
    <td>Knowledge Distillation</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/tensorflow/image_recognition/tensorflow_models/distillation">pb</a></td>
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
    <td>*ResNet18</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>*ResNet18</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/qat/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/qat/fx">fx</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/qat/eager/distributed">distributed</a></td>
  </tr>
  <tr>
    <td>*ResNet50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>*ResNet50</td>
    <td>Image Recognition</td>
    <td>Quantization-Aware Training</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/qat/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/qat/fx">fx</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/qat/eager/distributed">distributed</a></td>
  </tr>
  <tr>
    <td>ResNeXt101_32x16d_wsl</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>*ResNeXt101_32x8d</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>*Se_ResNeXt50_32x4d</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/se_resnext/quantization/ptq/eager">eager</a></td>
  </tr>
  <tr>
    <td>*Inception V3</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>*MobileNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/eager">eager</a> / <a href="./pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx">fx</a></td>
  </tr>
  <tr>
    <td>*PeleeNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/peleenet/quantization/ptq/eager">eager</a></td>
  </tr>
  <tr>
    <td>*ResNeSt50</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/resnest/quantization/ptq/eager">eager</a></td>
  </tr>
  <tr>
    <td>*3D-UNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/image_recognition/3d-unet/quantization/ptq/eager">eager</a></td>
  </tr>
  <tr>
    <td>SSD ResNet34</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/object_detection/ssd_resnet34/quantization/ptq/fx">fx</a> / <a href="./pytorch/object_detection/ssd_resnet34/quantization/ptq/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>*Mask R-CNN</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/object_detection/maskrcnn/quantization/ptq/fx">fx</a></td>
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
    <td>*RNN-T</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/speech_recognition/rnnt/quantization/ptq_dynamic/eager">eager</a> </td>
  </tr>
  <tr>
    <td>*Wav2Vec2</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/speech_recognition/torchaudio_models/quantization/ptq_dynamic/eager">eager</a> </td>
  </tr>
  <tr>
    <td>*HuBERT</td>
    <td>Speech Recognition</td>
    <td>Post-Training Dynamic /Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/speech_recognition/torchaudio_models/quantization/ptq_dynamic/eager">eager</a> / <a href="./pytorch/speech_recognition/torchaudio_models/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>*BlendCNN</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/blendcnn/quantization/ptq/eager">eager</a></td>
  </tr>
  <tr>
    <td>bert-large-uncased-whole-word-masking-finetuned-squad</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-classification/quantization/ptq_static/fx">fx</a> / <a href="./pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex">ipex</a></td>
  </tr>
    <tr>
    <td>distilbert-base-uncased-distilled-squad</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex">ipex</a></td>
  </tr>
  <tr>
    <td>*t5-small</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/eager">eager</a></td>
  </tr>
  <tr>
    <td>*Helsinki-NLP/opus-mt-en-ro</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/translation/quantization/ptq_dynamic/eager">eager</a></td>
  </tr>
  <tr>
    <td>*lvwerra/pegasus-samsum</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/summarization/quantization/ptq_dynamic/eager">eager</a></td>
  </tr>
  <tr>
    <td>GPTJ</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/language-modeling/quantization/ptq_static/fx">fx</a></td>
  </tr>
  <tr>
    <td>SD Diffusion</td>
    <td>Text to Image</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./pytorch/nlp/huggingface_models/text-to-image/quantization">fx</a></td>
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
    <td>YOLOv5s6</td>
    <td>Object Detection</td>
    <td>Structured (4x1, 2in4), Unstructured</td>
    <td>Snip-momentum</td>
    <td><a href="./pytorch/object_detection/yolo_v5/pruning/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet18</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/pruning/magnitude/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet34</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/pruning/magnitude/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/pruning/magnitude/eager">eager</a></td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>Image Recognition</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="./pytorch/image_recognition/torchvision_models/pruning/magnitude/eager">eager</a></td>
  </tr>
  <tr>
    <td>*BERT large</td>
    <td>Natural Language Processing</td>
    <td>Structured (2x1)</td>
    <td>Group Lasso</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager">eager</a></td>
  </tr>

  <tr>
    <td>*Intel/bert-base-uncased-sparse-70-unstructured</td>
    <td>Natural Language Processing (question-answering)</td>
    <td>Unstructured</td>
    <td>Prune once for all</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/question-answering/pruning/pattern_lock/eager">eager</a></td>
  </tr>
  <tr>
    <td>*bert-base-uncased</td>
    <td>Natural Language Processing</td>
    <td>Structured (Filter/Channel-wise)</td>
    <td>Gradient Sensitivity</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/text-classification/pruning/gradient_sensitivity/eager">eager</a></td>
  </tr>
  <tr>
    <td>*DistilBERT</td>
    <td>Natural Language Processing</td>
    <td>Unstructured</td>
    <td>Magnitude</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/text-classification/pruning/magnitude/eager">eager</a></td>
  </tr>
  <tr>
    <td>*Intel/bert-base-uncased-sparse-70-unstructured</td>
    <td>Natural Language Processing (text-classification)</td>
    <td>Unstructured</td>
    <td>Prune once for all</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/text-classification/pruning/pattern_lock/eager">eager</a></td>
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
    <td><a href="./pytorch/image_recognition/torchvision_models/optimization_pipeline/prune_and_ptq/eager">link</a></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>Image Recognition</td>
    <td>One-shot: QAT during Pruning<br></td>
    <td><a href="./pytorch/image_recognition/torchvision_models/optimization_pipeline/qat_during_prune/eager">link</a></td>
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
  <tr>
    <td>BERT-mini</td>
    <td>Natural Language Processing (text-classification)</td>
    <td>One-shot: Pruning, Distillation<br></td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/pytorch/nlp/huggingface_models/text-classification/optimization_pipeline/prune_once_for_all/fx">link</a></td>
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
    <td>*ResNet50 V1.5</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/resnet50/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/resnet50/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*ResNet50 V1.5 MLPerf</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/resnet50/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/resnet50/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*VGG16</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/vgg16/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/vgg16/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*MobileNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/mobilenet_v2/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/mobilenet_v2/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*MobileNet V3 MLPerf</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/mobilenet_v3/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/mobilenet_v3/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*AlexNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/alexnet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/alexnet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*CaffeNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/caffenet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/caffenet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*DenseNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/densenet/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*EfficientNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/efficientnet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/efficientnet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>FCN</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/fcn/quantization/ptq">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/fcn/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*GoogleNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/googlenet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/googlenet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*Inception V1</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/inception/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/inception/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*MNIST</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/mnist/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*MobileNet V2 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/mobilenet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/mobilenet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/resnet50/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/resnet50/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*ShuffleNet V2</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/shufflenet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/shufflenet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>SqueezeNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/image_recognition/onnx_model_zoo/squeezenet/quantization/ptq">qlinearops</a> / <a href="./onnxrt/image_recognition/onnx_model_zoo/squeezenet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*VGG16 (ONNX Model Zoo)</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/vgg16/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/vgg16/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*ZFNet</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/zfnet/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/image_recognition/onnx_model_zoo/zfnet/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>ArcFace</td>
    <td>Image Recognition</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/arcface/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*BERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/bert/quantization/ptq">integerops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/bert/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*BERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/bert/quantization/ptq">integerops</a></td>
  </tr>
  <tr>
    <td>*DistilBERT base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/distilbert/quantization/ptq">integerops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/distilbert/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*Mobile bert MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/mobilebert/quantization/ptq">integerops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/mobilebert/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*Roberta base MRPC</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/roberta/quantization/ptq">integerops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/nlp/roberta/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>BERT SQuAD</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/bert-squad/quantization/ptq">integerops</a> / <a href="./onnxrt/nlp/onnx_model_zoo/bert-squad/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>GPT2 lm head WikiText</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/gpt2/quantization/ptq">integerops</a></td>
  </tr>
  <tr>
    <td>MobileBERT SQuAD MLPerf</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/mobilebert/quantization/ptq">integerops</a> / <a href="./onnxrt/nlp/onnx_model_zoo/mobilebert/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>BiDAF</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic Quantization</td>
    <td><a href="./onnxrt/nlp/onnx_model_zoo/BiDAF/quantization/ptq">integerops</a></td>
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
    <td><a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
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
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2 (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td> 
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
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
        <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/text_classification/quantization/ptq_static">qlinearops</a>
    </td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD (HuggingFace)</td>
    <td>Natural Language Processing</td>
    <td>Post-Training Dynamic / Static Quantization</td>
    <td><a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_dynamic">integerops</a> / <a href="./onnxrt/nlp/huggingface_model/question_answering/quantization/ptq_static">qlinearops</a></td>
  </tr>
  <tr>
    <td>*SSD MobileNet V1</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/ssd_mobilenet_v1/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/ssd_mobilenet_v1/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*SSD MobileNet V2</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/ssd_mobilenet_v2/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/ssd_mobilenet_v2/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/ssd_mobilenet_v1/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/ssd_mobilenet_v1/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>DUC</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/object_detection/onnx_model_zoo/DUC/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*Faster R-CNN</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/faster_rcnn/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/faster_rcnn/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*Mask R-CNN</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/mask_rcnn/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/mask_rcnn/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*SSD</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/ssd/quantization/ptq">qlinearops</a> / <a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/ssd/quantization/ptq">qdq</a></td>
  </tr>
  <tr>
    <td>*Tiny YOLOv3</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/tiny_yolov3/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*YOLOv3</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/yolov3/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>*YOLOv4</td>
    <td>Object Detection</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="https://github.com/intel/neural-compressor/tree/old_api_examples/examples/onnxrt/object_detection/onnx_model_zoo/yolov4/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>Emotion FERPlus</td>
    <td>Body Analysis</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/emotion_ferplus/quantization/ptq">qlinearops</a></td>
  </tr>
  <tr>
    <td>Ultra Face</td>
    <td>Body Analysis</td>
    <td>Post-Training Static Quantization</td>
    <td><a href="./onnxrt/body_analysis/onnx_model_zoo/ultraface/quantization/ptq">qlinearops</a></td>
  </tr>
</tbody>
</table>


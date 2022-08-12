Validated Models
===
## Validated MLPerf Models
<table class="docutils">
<thead>
  <tr>
    <th>Model</th>
    <th>Framework</th>
    <th>Support</th>
    <th>Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">ResNet50 V1.5</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="../examples/tensorflow/image_recognition/tensorflow_models/quantization/ptq">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/ipex">Link</a></td>
  </tr>
  <tr>
    <td>DLRM</td>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/recommendation/dlrm/quantization/ptq">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2">BERT large</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="../examples/tensorflow/nlp/bert_large_squad/quantization/ptq">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/nlp/huggingface_models/question-answering/quantization/ptq_static/ipex">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2">SSD ResNet34</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="../examples/tensorflow/object_detection/tensorflow_models/quantization/ptq">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/object_detection/ssd_resnet34/quantization">Link</a></td>
  </tr>
  <tr>
    <td>RNN-T</td>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/speech_recognition/rnnt/quantization">Link</a></td>    
  </tr>
  <tr>
    <td rowspan="2">3D-UNet</td>
    <td>TensorFlow</td>
    <td>Yes</td>
    <td><a href="../examples/tensorflow/semantic_image_segmentation/3dunet-mlperf/quantization">Link</a></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/image_recognition/3d-unet/quantization">Link</a></td>
  </tr>
</tbody>
</table>

## Validated Quantization Examples

Performance results test on ​​07/26/2022 with Intel Xeon Platinum 8380 Scalable processor, using 1 socket, 4 cores/instance, 10 instances and batch size 1. 

Performance varies by use, configuration and other factors. See [platform configuration](./platform_configuration.md) for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks

### TensorFlow models with Intel TensorFlow 2.9.1

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput(samples/sec) <br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-7zrl">BERT large SQuAD</td>
    <td class="tg-8d8j">92.43</td>
    <td class="tg-8d8j">92.99</td>
    <td class="tg-8d8j">-0.60%</td>
    <td class="tg-8d8j">25.37</td>
    <td class="tg-8d8j">12.55</td>
    <td class="tg-8d8j">2.02x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
    <td class="tg-7zrl">DenseNet121</td>
    <td class="tg-8d8j">73.57%</td>
    <td class="tg-8d8j">72.89%</td>
    <td class="tg-8d8j">0.93%</td>
    <td class="tg-8d8j">368.05</td>
    <td class="tg-8d8j">328.84</td>
    <td class="tg-8d8j">1.12x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">DenseNet161</td>
    <td class="tg-8d8j">76.24%</td>
    <td class="tg-8d8j">76.29%</td>
    <td class="tg-8d8j">-0.07%</td>
    <td class="tg-8d8j">219.08</td>
    <td class="tg-8d8j">179.2</td>
    <td class="tg-8d8j">1.22x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">DenseNet169</td>
    <td class="tg-8d8j">74.40%</td>
    <td class="tg-8d8j">74.65%</td>
    <td class="tg-8d8j">-0.33%</td>
    <td class="tg-8d8j">295.26</td>
    <td class="tg-8d8j">260.27</td>
    <td class="tg-8d8j">1.13x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN Inception ResNet V2</td>
    <td class="tg-8d8j">37.98%</td>
    <td class="tg-8d8j">38.33%</td>
    <td class="tg-8d8j">-0.91%</td>
    <td class="tg-8d8j">3.97</td>
    <td class="tg-8d8j">2.34</td>
    <td class="tg-8d8j">1.70x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN Inception ResNet V2 </td>
    <td class="tg-8d8j">37.84%</td>
    <td class="tg-8d8j">38.33%</td>
    <td class="tg-8d8j">-1.28%</td>
    <td class="tg-8d8j">4</td>
    <td class="tg-8d8j">2.32</td>
    <td class="tg-8d8j">1.73x</td>
    <td class="tg-7zrl">SavedModel</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN ResNet101</td>
    <td class="tg-8d8j">30.28%</td>
    <td class="tg-8d8j">30.39%</td>
    <td class="tg-8d8j">-0.36%</td>
    <td class="tg-8d8j">70.32</td>
    <td class="tg-8d8j">20.19</td>
    <td class="tg-8d8j">3.48x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN ResNet101</td>
    <td class="tg-8d8j">30.37%</td>
    <td class="tg-8d8j">30.39%</td>
    <td class="tg-8d8j">-0.07%</td>
    <td class="tg-8d8j">70.3</td>
    <td class="tg-8d8j">17.1</td>
    <td class="tg-8d8j">4.11x</td>
    <td class="tg-7zrl">SavedModel</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN ResNet50</td>
    <td class="tg-8d8j">26.57%</td>
    <td class="tg-8d8j">26.59%</td>
    <td class="tg-8d8j">-0.08%</td>
    <td class="tg-8d8j">83.32</td>
    <td class="tg-8d8j">24.38</td>
    <td class="tg-8d8j">3.42x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception ResNet V2</td>
    <td class="tg-8d8j">80.28%</td>
    <td class="tg-8d8j">80.40%</td>
    <td class="tg-8d8j">-0.15%</td>
    <td class="tg-8d8j">287.05</td>
    <td class="tg-8d8j">136.78</td>
    <td class="tg-8d8j">2.10x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception V1</td>
    <td class="tg-8d8j">70.48%</td>
    <td class="tg-8d8j">69.74%</td>
    <td class="tg-8d8j">1.06%</td>
    <td class="tg-8d8j">2208.41</td>
    <td class="tg-8d8j">977.76</td>
    <td class="tg-8d8j">2.26x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception V2</td>
    <td class="tg-8d8j">74.36%</td>
    <td class="tg-8d8j">73.97%</td>
    <td class="tg-8d8j">0.53%</td>
    <td class="tg-8d8j">1847.6</td>
    <td class="tg-8d8j">828.03</td>
    <td class="tg-8d8j">2.23x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception V3</td>
    <td class="tg-8d8j">76.71%</td>
    <td class="tg-8d8j">76.75%</td>
    <td class="tg-8d8j">-0.05%</td>
    <td class="tg-8d8j">1036.18</td>
    <td class="tg-8d8j">373.61</td>
    <td class="tg-8d8j">2.77x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception V4</td>
    <td class="tg-8d8j">80.20%</td>
    <td class="tg-8d8j">80.27%</td>
    <td class="tg-8d8j">-0.09%</td>
    <td class="tg-8d8j">592.46</td>
    <td class="tg-8d8j">206.96</td>
    <td class="tg-8d8j">2.86x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Mask R-CNN Inception V2</td>
    <td class="tg-8d8j">28.53%</td>
    <td class="tg-8d8j">28.73%</td>
    <td class="tg-8d8j">-0.70%</td>
    <td class="tg-8d8j">132.07</td>
    <td class="tg-8d8j">51</td>
    <td class="tg-8d8j">2.59x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Mask R-CNN Inception V2 </td>
    <td class="tg-8d8j">28.53%</td>
    <td class="tg-8d8j">28.73%</td>
    <td class="tg-8d8j">-0.70%</td>
    <td class="tg-8d8j">132.41</td>
    <td class="tg-8d8j">50.94</td>
    <td class="tg-8d8j">2.60x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-8d8j">MobileNet V1</td>
    <td class="tg-8d8j">71.79%</td>
    <td class="tg-8d8j">70.96%</td>
    <td class="tg-8d8j">1.17%</td>
    <td class="tg-8d8j">3603.94</td>
    <td class="tg-8d8j">1304.58</td>
    <td class="tg-8d8j">2.76x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">MobileNet V2</td>
    <td class="tg-8d8j">71.89%</td>
    <td class="tg-8d8j">71.76%</td>
    <td class="tg-8d8j">0.18%</td>
    <td class="tg-8d8j">2433.87</td>
    <td class="tg-8d8j">1446.1</td>
    <td class="tg-8d8j">1.68x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet101</td>
    <td class="tg-8d8j">77.50%</td>
    <td class="tg-8d8j">76.45%</td>
    <td class="tg-8d8j">1.37%</td>
    <td class="tg-8d8j">874.26</td>
    <td class="tg-8d8j">356.84</td>
    <td class="tg-8d8j">2.45x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet50 Fashion</td>
    <td class="tg-8d8j">78.06%</td>
    <td class="tg-8d8j">78.12%</td>
    <td class="tg-8d8j">-0.08%</td>
    <td class="tg-8d8j">3776.14</td>
    <td class="tg-8d8j">2160.52</td>
    <td class="tg-8d8j">1.75x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet50 V1.0</td>
    <td class="tg-8d8j">74.11%</td>
    <td class="tg-8d8j">74.27%</td>
    <td class="tg-8d8j">-0.22%</td>
    <td class="tg-8d8j">1511.74</td>
    <td class="tg-8d8j">459.43</td>
    <td class="tg-8d8j">3.29x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet50 V1.5</td>
    <td class="tg-8d8j">76.22%</td>
    <td class="tg-8d8j">76.46%</td>
    <td class="tg-8d8j">-0.31%</td>
    <td class="tg-8d8j">1355.03</td>
    <td class="tg-8d8j">423.41</td>
    <td class="tg-8d8j">3.20x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet V2 101</td>
    <td class="tg-8d8j">72.67%</td>
    <td class="tg-8d8j">71.87%</td>
    <td class="tg-8d8j">1.11%</td>
    <td class="tg-8d8j">436.34</td>
    <td class="tg-8d8j">323.15</td>
    <td class="tg-8d8j">1.35x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet V2 152</td>
    <td class="tg-8d8j">73.03%</td>
    <td class="tg-8d8j">72.37%</td>
    <td class="tg-8d8j">0.91%</td>
    <td class="tg-8d8j">311.93</td>
    <td class="tg-8d8j">222.83</td>
    <td class="tg-8d8j">1.40x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet V2 50</td>
    <td class="tg-8d8j">70.33%</td>
    <td class="tg-8d8j">69.64%</td>
    <td class="tg-8d8j">0.99%</td>
    <td class="tg-8d8j">766.83</td>
    <td class="tg-8d8j">574.76</td>
    <td class="tg-8d8j">1.33x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SSD MobileNet V1</td>
    <td class="tg-8d8j">22.97%</td>
    <td class="tg-8d8j">23.13%</td>
    <td class="tg-8d8j">-0.69%</td>
    <td class="tg-8d8j">959.72</td>
    <td class="tg-8d8j">586.21</td>
    <td class="tg-8d8j">1.64x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SSD MobileNet V1</td>
    <td class="tg-8d8j">22.99%</td>
    <td class="tg-8d8j">23.13%</td>
    <td class="tg-8d8j">-0.61%</td>
    <td class="tg-8d8j">953.4</td>
    <td class="tg-8d8j">412.06</td>
    <td class="tg-8d8j">2.31x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SSD ResNet34</td>
    <td class="tg-8d8j">21.69%</td>
    <td class="tg-8d8j">22.09%</td>
    <td class="tg-8d8j">-1.81%</td>
    <td class="tg-8d8j">44.53</td>
    <td class="tg-8d8j">11.86</td>
    <td class="tg-8d8j">3.75x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SSD ResNet50 V1</td>
    <td class="tg-8d8j">37.86%</td>
    <td class="tg-8d8j">38.00%</td>
    <td class="tg-8d8j">-0.37%</td>
    <td class="tg-8d8j">69.09</td>
    <td class="tg-8d8j">25.93</td>
    <td class="tg-8d8j">2.66x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SSD ResNet50 V1</td>
    <td class="tg-8d8j">37.81%</td>
    <td class="tg-8d8j">38.00%</td>
    <td class="tg-8d8j">-0.50%</td>
    <td class="tg-8d8j">69.02</td>
    <td class="tg-8d8j">21.06</td>
    <td class="tg-8d8j">3.28x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-8d8j">VGG16</td>
    <td class="tg-8d8j">72.66%</td>
    <td class="tg-8d8j">70.89%</td>
    <td class="tg-8d8j">2.50%</td>
    <td class="tg-8d8j">660.05</td>
    <td class="tg-8d8j">177.23</td>
    <td class="tg-8d8j">3.72x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">VGG19</td>
    <td class="tg-8d8j">72.72%</td>
    <td class="tg-8d8j">71.01%</td>
    <td class="tg-8d8j">2.41%</td>
    <td class="tg-8d8j">560.39</td>
    <td class="tg-8d8j">147.27</td>
    <td class="tg-8d8j">3.81x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Wide &amp; Deep</td>
    <td class="tg-8d8j">77.62%</td>
    <td class="tg-8d8j">77.67%</td>
    <td class="tg-8d8j">-0.07%</td>
    <td class="tg-8d8j">23329.18</td>
    <td class="tg-8d8j">20930.18</td>
    <td class="tg-8d8j">1.11x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.12.0+cpu in PTQ mode

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput (samples/sec) <br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-8d8j">ALBERT base MRPC</td>
    <td class="tg-8d8j">88.85%</td>
    <td class="tg-8d8j">88.50%</td>
    <td class="tg-8d8j">0.40%</td>
    <td class="tg-8d8j">34.46</td>
    <td class="tg-8d8j">26.88</td>
    <td class="tg-8d8j">1.28x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Barthez MRPC</td>
    <td class="tg-8d8j">83.92%</td>
    <td class="tg-8d8j">83.81%</td>
    <td class="tg-8d8j">0.14%</td>
    <td class="tg-8d8j">161.06</td>
    <td class="tg-8d8j">89.61</td>
    <td class="tg-8d8j">1.80x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT base COLA</td>
    <td class="tg-8d8j">58.80%</td>
    <td class="tg-8d8j">58.84%</td>
    <td class="tg-8d8j">-0.07%</td>
    <td class="tg-8d8j">262.88</td>
    <td class="tg-8d8j">125.63</td>
    <td class="tg-8d8j">2.09x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT base MRPC</td>
    <td class="tg-8d8j">89.90%</td>
    <td class="tg-8d8j">90.69%</td>
    <td class="tg-8d8j">-0.88%</td>
    <td class="tg-8d8j">244.27</td>
    <td class="tg-8d8j">125.28</td>
    <td class="tg-8d8j">1.95x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT base RTE</td>
    <td class="tg-8d8j">69.31%</td>
    <td class="tg-8d8j">69.68%</td>
    <td class="tg-8d8j">-0.52%</td>
    <td class="tg-8d8j">259.21</td>
    <td class="tg-8d8j">125.72</td>
    <td class="tg-8d8j">2.06x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT base SST2</td>
    <td class="tg-8d8j">91.06%</td>
    <td class="tg-8d8j">91.86%</td>
    <td class="tg-8d8j">-0.87%</td>
    <td class="tg-8d8j">262.73</td>
    <td class="tg-8d8j">125.69</td>
    <td class="tg-8d8j">2.09x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT base STSB</td>
    <td class="tg-8d8j">89.10%</td>
    <td class="tg-8d8j">89.75%</td>
    <td class="tg-8d8j">-0.72%</td>
    <td class="tg-8d8j">254.36</td>
    <td class="tg-8d8j">125.9</td>
    <td class="tg-8d8j">2.02x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT large COLA</td>
    <td class="tg-8d8j">64.12%</td>
    <td class="tg-8d8j">62.57%</td>
    <td class="tg-8d8j">2.48%</td>
    <td class="tg-8d8j">89.36</td>
    <td class="tg-8d8j">36.47</td>
    <td class="tg-8d8j">2.45x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT large MRPC</td>
    <td class="tg-8d8j">89.50%</td>
    <td class="tg-8d8j">90.38%</td>
    <td class="tg-8d8j">-0.97%</td>
    <td class="tg-8d8j">88.92</td>
    <td class="tg-8d8j">36.55</td>
    <td class="tg-8d8j">2.43x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">BERT large QNLI</td>
    <td class="tg-8d8j">90.90%</td>
    <td class="tg-8d8j">91.82%</td>
    <td class="tg-8d8j">-1.00%</td>
    <td class="tg-8d8j">90.39</td>
    <td class="tg-8d8j">36.63</td>
    <td class="tg-8d8j">2.47x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">CamemBERT base MRPC</td>
    <td class="tg-8d8j">86.70%</td>
    <td class="tg-8d8j">86.82%</td>
    <td class="tg-8d8j">-0.14%</td>
    <td class="tg-8d8j">236.6</td>
    <td class="tg-8d8j">121.81</td>
    <td class="tg-8d8j">1.94x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Deberta MRPC</td>
    <td class="tg-8d8j">90.88%</td>
    <td class="tg-8d8j">90.91%</td>
    <td class="tg-8d8j">-0.04%</td>
    <td class="tg-8d8j">149.76</td>
    <td class="tg-8d8j">84.72</td>
    <td class="tg-8d8j">1.77x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">DistilBERT base MRPC</td>
    <td class="tg-8d8j">88.23%</td>
    <td class="tg-8d8j">89.16%</td>
    <td class="tg-8d8j">-1.05%</td>
    <td class="tg-8d8j">426.4</td>
    <td class="tg-8d8j">246.13</td>
    <td class="tg-8d8j">1.73x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">FlauBERT MRPC</td>
    <td class="tg-8d8j">79.87%</td>
    <td class="tg-8d8j">80.19%</td>
    <td class="tg-8d8j">-0.40%</td>
    <td class="tg-8d8j">675.82</td>
    <td class="tg-8d8j">437.72</td>
    <td class="tg-8d8j">1.54x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Inception V3</td>
    <td class="tg-8d8j">69.43%</td>
    <td class="tg-8d8j">69.52%</td>
    <td class="tg-8d8j">-0.13%</td>
    <td class="tg-8d8j">490.32</td>
    <td class="tg-8d8j">209.87</td>
    <td class="tg-8d8j">2.34x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Longformer MRPC</td>
    <td class="tg-8d8j">91.01%</td>
    <td class="tg-8d8j">91.46%</td>
    <td class="tg-8d8j">-0.49%</td>
    <td class="tg-8d8j">20.36</td>
    <td class="tg-8d8j">16.65</td>
    <td class="tg-8d8j">1.22x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mBart WNLI</td>
    <td class="tg-8d8j">56.34%</td>
    <td class="tg-8d8j">56.34%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">66.23</td>
    <td class="tg-8d8j">30.86</td>
    <td class="tg-8d8j">2.15x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">lvwerra/pegasus-samsum</td>
    <td class="tg-8d8j">42.39</td>
    <td class="tg-8d8j">42.67</td>
    <td class="tg-8d8j">-0.67%</td>
    <td class="tg-8d8j">3.86</td>
    <td class="tg-8d8j">1.14</td>
    <td class="tg-8d8j">3.38x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">PeleeNet</td>
    <td class="tg-8d8j">71.64%</td>
    <td class="tg-8d8j">72.10%</td>
    <td class="tg-8d8j">-0.64%</td>
    <td class="tg-8d8j">511.56</td>
    <td class="tg-8d8j">387.9</td>
    <td class="tg-8d8j">1.32x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet18 </td>
    <td class="tg-8d8j">69.57%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.27%</td>
    <td class="tg-8d8j">823.22</td>
    <td class="tg-8d8j">386.93</td>
    <td class="tg-8d8j">2.13x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet18 </td>
    <td class="tg-8d8j">69.57%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.28%</td>
    <td class="tg-8d8j">816.8</td>
    <td class="tg-8d8j">385.23</td>
    <td class="tg-8d8j">2.12x</td>
    <td class="tg-8d8j">fx</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet50</td>
    <td class="tg-8d8j">75.98%</td>
    <td class="tg-8d8j">76.15%</td>
    <td class="tg-8d8j">-0.21%</td>
    <td class="tg-8d8j">515.14</td>
    <td class="tg-8d8j">204</td>
    <td class="tg-8d8j">2.53x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNeXt101_32x8d</td>
    <td class="tg-8d8j">79.08%</td>
    <td class="tg-8d8j">79.31%</td>
    <td class="tg-8d8j">-0.29%</td>
    <td class="tg-8d8j">210.39</td>
    <td class="tg-8d8j">74.87</td>
    <td class="tg-8d8j">2.81x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">RNNT</td>
    <td class="tg-8d8j">92.48</td>
    <td class="tg-8d8j">92.55</td>
    <td class="tg-8d8j">-0.08%</td>
    <td class="tg-8d8j">74.17</td>
    <td class="tg-8d8j">20.38</td>
    <td class="tg-8d8j">3.64x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Roberta Base MRPC</td>
    <td class="tg-8d8j">88.25%</td>
    <td class="tg-8d8j">88.18%</td>
    <td class="tg-8d8j">0.08%</td>
    <td class="tg-8d8j">245.05</td>
    <td class="tg-8d8j">123.53</td>
    <td class="tg-8d8j">1.98x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Se_ResNeXt50_32x4d</td>
    <td class="tg-8d8j">78.98%</td>
    <td class="tg-8d8j">79.08%</td>
    <td class="tg-8d8j">-0.13%</td>
    <td class="tg-8d8j">370.11</td>
    <td class="tg-8d8j">172.45</td>
    <td class="tg-8d8j">2.15x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">SqueezeBERT MRPC</td>
    <td class="tg-8d8j">86.87%</td>
    <td class="tg-8d8j">87.65%</td>
    <td class="tg-8d8j">-0.89%</td>
    <td class="tg-8d8j">241.25</td>
    <td class="tg-8d8j">206.03</td>
    <td class="tg-8d8j">1.17x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">Transfo-xl MRPC</td>
    <td class="tg-8d8j">81.97%</td>
    <td class="tg-8d8j">81.20%</td>
    <td class="tg-8d8j">0.94%</td>
    <td class="tg-8d8j">11.2</td>
    <td class="tg-8d8j">8.31</td>
    <td class="tg-8d8j">1.35x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-8d8j">xlm-roberta-base_MRPC</td>
    <td class="tg-8d8j">88.03%</td>
    <td class="tg-8d8j">88.62%</td>
    <td class="tg-8d8j">-0.67%</td>
    <td class="tg-8d8j">140.58</td>
    <td class="tg-8d8j">122.29</td>
    <td class="tg-8d8j">1.15x</td>
    <td class="tg-8d8j">eager</td>
  </tr>
  <tr>
    <td class="tg-6985">YOLOv3</td>
    <td class="tg-nrix">24.60%</td>
    <td class="tg-nrix">24.54%</td>
    <td class="tg-nrix">0.21%</td>
    <td class="tg-nrix">110.54</td>
    <td class="tg-nrix">39.46</td>
    <td class="tg-nrix">2.80x</td>
    <td class="tg-nrix">eager</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.12.0+cpu in QAT mode

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput (samples/sec) <br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-6985">ResNet18</td>
    <td class="tg-nrix">69.84%</td>
    <td class="tg-nrix">69.76%</td>
    <td class="tg-nrix">0.11%</td>
    <td class="tg-nrix">805.76</td>
    <td class="tg-nrix">389.14</td>
    <td class="tg-nrix">2.07x</td>
    <td class="tg-nrix">eager</td>
  </tr>
  <tr>
    <td class="tg-6985">ResNet18</td>
    <td class="tg-nrix">69.74%</td>
    <td class="tg-nrix">69.76%</td>
    <td class="tg-nrix">-0.03%</td>
    <td class="tg-nrix">822.99</td>
    <td class="tg-nrix">391.82</td>
    <td class="tg-nrix">2.10x</td>
    <td class="tg-nrix">fx</td>
  </tr>
  <tr>
    <td class="tg-nrix">BERT base MRPC QAT</td>
    <td class="tg-nrix">89.70%</td>
    <td class="tg-nrix">89.50%</td>
    <td class="tg-nrix">0.22%</td>
    <td class="tg-nrix">173.83</td>
    <td class="tg-nrix">107.22</td>
    <td class="tg-nrix">1.62x</td>
    <td class="tg-nrix">fx</td>
  </tr>
  <tr>
    <td class="tg-6985">ResNet50</td>
    <td class="tg-nrix">76.05%</td>
    <td class="tg-nrix">76.15%</td>
    <td class="tg-nrix">-0.13%</td>
    <td class="tg-nrix">500.54</td>
    <td class="tg-nrix">195.5</td>
    <td class="tg-nrix">2.56x</td>
    <td class="tg-nrix">eager</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.12.0+cpu IPEX

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput (samples/sec) <br>1s 10ins 4c/ins bs=8<br></th>
    <th class="tg-7zrl" colspan="3">Performance<br>Throughput (samples/sec) <br>1s 10ins 4c/ins bs=1<br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
    <th class="tg-7zrl">INT8</th>
    <th class="tg-7zrl">FP32</th>
    <th class="tg-7zrl">INT8/FP32</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-7zrl">SSD ResNet34</td>
    <td class="tg-7zrl">19.99%</td>
    <td class="tg-7zrl">20.00%</td>
    <td class="tg-7zrl">-0.06%</td>
    <td class="tg-7zrl">11.19</td>
    <td class="tg-7zrl">8.72</td>
    <td class="tg-7zrl">1.28x</td>
    <td class="tg-7zrl">11.63</td>
    <td class="tg-7zrl">9.82</td>
    <td class="tg-7zrl">1.18x</td>
    <td class="tg-7zrl">ipex</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.11.0+cpu IPEX

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput(samples/sec) <br>1s 10ins 4c/ins bs=64<br></th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput(samples/sec) <br>1s 10ins 4c/ins bs=1<br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">INT8/FP32</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-nrix">ResNet18</td>
    <td class="tg-nrix">69.48%</td>
    <td class="tg-nrix">69.76%</td>
    <td class="tg-nrix">-0.40%</td>
    <td class="tg-nrix">3968.05</td>
    <td class="tg-nrix">1182.36</td>
    <td class="tg-nrix">3.36x</td>
    <td class="tg-nrix">2995.8</td>
    <td class="tg-nrix">1140.63</td>
    <td class="tg-nrix">2.63x</td>
    <td class="tg-nrix">ipex</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNeXt101_32x16d_wsl</td>
    <td class="tg-nrix">84.26%</td>
    <td class="tg-nrix">84.17%</td>
    <td class="tg-nrix">0.11%</td>
    <td class="tg-nrix">195.72</td>
    <td class="tg-nrix">51.28</td>
    <td class="tg-nrix">3.82x</td>
    <td class="tg-nrix">178.01</td>
    <td class="tg-nrix">59.12</td>
    <td class="tg-nrix">3.01x</td>
    <td class="tg-nrix">ipex</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNet50</td>
    <td class="tg-nrix">76.07%</td>
    <td class="tg-nrix">76.15%</td>
    <td class="tg-nrix">-0.10%</td>
    <td class="tg-nrix">1731.8</td>
    <td class="tg-nrix">474.24</td>
    <td class="tg-nrix">3.65x</td>
    <td class="tg-nrix">1347.64</td>
    <td class="tg-nrix">518.39</td>
    <td class="tg-nrix">2.60x</td>
    <td class="tg-nrix">ipex</td>
  </tr>
</tbody>
</table>

### ONNX Models with ONNX Runtime 1.11.0

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput(samples/sec) <br></th>
    <th class="tg-nrix" rowspan="2">Example</th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-nrix">AlexNet </td>
    <td class="tg-nrix">54.74%</td>
    <td class="tg-nrix">54.79%</td>
    <td class="tg-nrix">-0.09%</td>
    <td class="tg-nrix">1498.07</td>
    <td class="tg-nrix">649.99</td>
    <td class="tg-nrix">2.30x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">BERT base MRPC DYNAMIC</td>
    <td class="tg-nrix">85.54%</td>
    <td class="tg-nrix">86.03%</td>
    <td class="tg-nrix">-0.57%</td>
    <td class="tg-nrix">381.45</td>
    <td class="tg-nrix">156.05</td>
    <td class="tg-nrix">2.44x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">BERT base MRPC STATIC</td>
    <td class="tg-nrix">85.29%</td>
    <td class="tg-nrix">86.03%</td>
    <td class="tg-nrix">-0.86%</td>
    <td class="tg-nrix">766.09</td>
    <td class="tg-nrix">316.6</td>
    <td class="tg-nrix">2.42x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">BERT SQuAD</td>
    <td class="tg-nrix">80.44</td>
    <td class="tg-nrix">80.67</td>
    <td class="tg-nrix">-0.29%</td>
    <td class="tg-nrix">116.88</td>
    <td class="tg-nrix">64.59</td>
    <td class="tg-nrix">1.81x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">BERT SQuAD</td>
    <td class="tg-nrix">80.44</td>
    <td class="tg-nrix">80.67</td>
    <td class="tg-nrix">-0.29%</td>
    <td class="tg-nrix">116.93</td>
    <td class="tg-nrix">64.64</td>
    <td class="tg-nrix">1.81x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">BiDAF</td>
    <td class="tg-nrix">65.92%</td>
    <td class="tg-nrix">66.08%</td>
    <td class="tg-nrix">-0.24%</td>
    <td class="tg-nrix">1468.58</td>
    <td class="tg-nrix">1406.21</td>
    <td class="tg-nrix">1.04x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">CaffeNet</td>
    <td class="tg-nrix">56.26%</td>
    <td class="tg-nrix">56.30%</td>
    <td class="tg-nrix">-0.07%</td>
    <td class="tg-nrix">2750.7</td>
    <td class="tg-nrix">812.73</td>
    <td class="tg-nrix">3.38x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">DistilBERT base MRPC</td>
    <td class="tg-nrix">84.56%</td>
    <td class="tg-nrix">84.56%</td>
    <td class="tg-nrix">0.00%</td>
    <td class="tg-nrix">1654.95</td>
    <td class="tg-nrix">595.32</td>
    <td class="tg-nrix">2.78x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">EfficientNet</td>
    <td class="tg-nrix">77.58%</td>
    <td class="tg-nrix">77.70%</td>
    <td class="tg-nrix">-0.15%</td>
    <td class="tg-nrix">2066.02</td>
    <td class="tg-nrix">1096.86</td>
    <td class="tg-nrix">1.88x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">FCN</td>
    <td class="tg-nrix">64.66%</td>
    <td class="tg-nrix">64.98%</td>
    <td class="tg-nrix">-0.49%</td>
    <td class="tg-nrix">15.13</td>
    <td class="tg-nrix">7.2</td>
    <td class="tg-nrix">2.10x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">GoogleNet</td>
    <td class="tg-nrix">67.67%</td>
    <td class="tg-nrix">67.79%</td>
    <td class="tg-nrix">-0.18%</td>
    <td class="tg-nrix">1174.98</td>
    <td class="tg-nrix">807.68</td>
    <td class="tg-nrix">1.45x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">Inception V1</td>
    <td class="tg-nrix">67.23%</td>
    <td class="tg-nrix">67.24%</td>
    <td class="tg-nrix">-0.01%</td>
    <td class="tg-nrix">1181.71</td>
    <td class="tg-nrix">831.01</td>
    <td class="tg-nrix">1.42x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">Mobile bert MRPC</td>
    <td class="tg-nrix">86.03%</td>
    <td class="tg-nrix">86.27%</td>
    <td class="tg-nrix">-0.28%</td>
    <td class="tg-nrix">774.96</td>
    <td class="tg-nrix">678.66</td>
    <td class="tg-nrix">1.14x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">MobileBERT SQuAD MLPerf</td>
    <td class="tg-nrix">89.84</td>
    <td class="tg-nrix">90.03</td>
    <td class="tg-nrix">-0.20%</td>
    <td class="tg-nrix">104.51</td>
    <td class="tg-nrix">94.88</td>
    <td class="tg-nrix">1.10x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">MobileNet V2</td>
    <td class="tg-nrix">65.47%</td>
    <td class="tg-nrix">66.89%</td>
    <td class="tg-nrix">-2.12%</td>
    <td class="tg-nrix">5172.04</td>
    <td class="tg-nrix">3312.76</td>
    <td class="tg-nrix">1.56x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">MobileNet V3 MLPerf</td>
    <td class="tg-nrix">75.59%</td>
    <td class="tg-nrix">75.74%</td>
    <td class="tg-nrix">-0.20%</td>
    <td class="tg-nrix">4168.8</td>
    <td class="tg-nrix">2146.59</td>
    <td class="tg-nrix">1.94x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNet50 v1.5 MLPerf</td>
    <td class="tg-nrix">76.13%</td>
    <td class="tg-nrix">76.46%</td>
    <td class="tg-nrix">-0.43%</td>
    <td class="tg-nrix">1154.73</td>
    <td class="tg-nrix">554.69</td>
    <td class="tg-nrix">2.08x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNet50 V1.5</td>
    <td class="tg-nrix">72.28%</td>
    <td class="tg-nrix">72.29%</td>
    <td class="tg-nrix">-0.01%</td>
    <td class="tg-nrix">1156.05</td>
    <td class="tg-nrix">555.72</td>
    <td class="tg-nrix">2.08x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td class="tg-nrix">74.76%</td>
    <td class="tg-nrix">74.99%</td>
    <td class="tg-nrix">-0.31%</td>
    <td class="tg-nrix">1347.89</td>
    <td class="tg-nrix">588.84</td>
    <td class="tg-nrix">2.29x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td class="tg-nrix">74.75%</td>
    <td class="tg-nrix">74.99%</td>
    <td class="tg-nrix">-0.32%</td>
    <td class="tg-nrix">840.87</td>
    <td class="tg-nrix">588.77</td>
    <td class="tg-nrix">1.43x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
  <tr>
    <td class="tg-nrix">Roberta Base MRPC</td>
    <td class="tg-nrix">90.44%</td>
    <td class="tg-nrix">89.95%</td>
    <td class="tg-nrix">0.54%</td>
    <td class="tg-nrix">811.74</td>
    <td class="tg-nrix">312.93</td>
    <td class="tg-nrix">2.59x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">Tiny YOLOv3</td>
    <td class="tg-nrix">12.08%</td>
    <td class="tg-nrix">12.43%</td>
    <td class="tg-nrix">-2.82%</td>
    <td class="tg-nrix">801.46</td>
    <td class="tg-nrix">653.42</td>
    <td class="tg-nrix">1.23x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">VGG16</td>
    <td class="tg-nrix">66.60%</td>
    <td class="tg-nrix">66.69%</td>
    <td class="tg-nrix">-0.13%</td>
    <td class="tg-nrix">312.98</td>
    <td class="tg-nrix">128.7</td>
    <td class="tg-nrix">2.43x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">VGG16 (ONNX Model Zoo)</td>
    <td class="tg-nrix">72.28%</td>
    <td class="tg-nrix">72.40%</td>
    <td class="tg-nrix">-0.17%</td>
    <td class="tg-nrix">450.47</td>
    <td class="tg-nrix">130.74</td>
    <td class="tg-nrix">3.45x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">YOLOv3</td>
    <td class="tg-nrix">26.88%</td>
    <td class="tg-nrix">28.74%</td>
    <td class="tg-nrix">-6.47%</td>
    <td class="tg-nrix">157.58</td>
    <td class="tg-nrix">66.62</td>
    <td class="tg-nrix">2.37x</td>
    <td class="tg-nrix">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-nrix">ZFNet</td>
    <td class="tg-nrix">55.89%</td>
    <td class="tg-nrix">55.96%</td>
    <td class="tg-nrix">-0.13%</td>
    <td class="tg-nrix">658.93</td>
    <td class="tg-nrix">359.42</td>
    <td class="tg-nrix">1.83x</td>
    <td class="tg-nrix">qdq</td>
  </tr>
</tbody>
</table>

### MXNet models with MXNet 1.7.0

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Model</th>
    <th class="tg-nrix" colspan="3">Accuracy</th>
    <th class="tg-nrix" colspan="3">Performance<br>Throughput(samples/sec) <br></th>
  </tr>
  <tr>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Acc Ratio[(INT8-FP32)/FP32]</th>
    <th class="tg-nrix">INT8</th>
    <th class="tg-nrix">FP32</th>
    <th class="tg-nrix">Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-7zrl">Inception V3</td>
    <td class="tg-nrix">77.80%</td>
    <td class="tg-nrix">77.65%</td>
    <td class="tg-nrix">0.20%</td>
    <td class="tg-nrix">922.38</td>
    <td class="tg-nrix">277.59</td>
    <td class="tg-nrix">3.32x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MobileNet V1</td>
    <td class="tg-nrix">71.60%</td>
    <td class="tg-nrix">72.23%</td>
    <td class="tg-nrix">-0.86%</td>
    <td class="tg-nrix">6614.69</td>
    <td class="tg-nrix">2560.42</td>
    <td class="tg-nrix">2.58x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MobileNet V3 MLPerf</td>
    <td class="tg-nrix">70.80%</td>
    <td class="tg-nrix">70.87%</td>
    <td class="tg-nrix">-0.10%</td>
    <td class="tg-nrix">5230.58</td>
    <td class="tg-nrix">2024.85</td>
    <td class="tg-nrix">2.58x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ResNet v1 152</td>
    <td class="tg-nrix">78.28%</td>
    <td class="tg-nrix">78.54%</td>
    <td class="tg-nrix">-0.33%</td>
    <td class="tg-nrix">578.27</td>
    <td class="tg-nrix">156.38</td>
    <td class="tg-nrix">3.70x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNet50 V1.0</td>
    <td class="tg-nrix">75.91%</td>
    <td class="tg-nrix">76.33%</td>
    <td class="tg-nrix">-0.55%</td>
    <td class="tg-nrix">1571.33</td>
    <td class="tg-nrix">429.53</td>
    <td class="tg-nrix">3.66x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SqueezeNet</td>
    <td class="tg-nrix">56.80%</td>
    <td class="tg-nrix">56.97%</td>
    <td class="tg-nrix">-0.28%</td>
    <td class="tg-nrix">4712.15</td>
    <td class="tg-nrix">1323.68</td>
    <td class="tg-nrix">3.56x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V1</td>
    <td class="tg-nrix">74.94%</td>
    <td class="tg-nrix">75.54%</td>
    <td class="tg-nrix">-0.79%</td>
    <td class="tg-nrix">768.59</td>
    <td class="tg-nrix">191.55</td>
    <td class="tg-nrix">4.01x</td>
  </tr>
</tbody>
</table>

## Validated Pruning Examples
<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">FP32 Baseline</th>
    <th colspan="3">Gradient Sensitivity with 20% Sparsity</th>
    <th colspan="3">+ONNX Dynamic Quantization on Pruned Model</th>
  </tr>
  <tr>
    <td>Accuracy%</td>
    <td>Drop</td>
    <td>Perf Gain (sample/s)</td>
    <td>Accuracy%</td>
    <td>Drop</td>
    <td>Perf Gain (sample/s)</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SST-2</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>accuracy = 92.32</td>
    <td>accuracy = 91.97</td>
    <td>-0.38</td>
    <td>1.30x</td>
    <td>accuracy = 92.20</td>
    <td>-0.13</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>[accuracy, f1] = [91.10, 88.05]</td>
    <td>[accuracy, f1] = [89.97, 86.54]</td>
    <td>[-1.24, -1.71]</td>
    <td>1.32x</td>
    <td>[accuracy, f1] = [89.75, 86.60]</td>
    <td>[-1.48, -1.65]</td>
    <td>1.81x</td>
  </tr>
</tbody>
</table>

<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">FP32 Baseline</th>
    <th colspan="2">Pattern Lock on 70% Unstructured Sparsity</th>
    <th colspan="2">Pattern Lock on 50% 1:2 Structured Sparsity</th>
  </tr>
  <tr>
    <td>Accuracy%</td>
    <td>Drop</td>
    <td>Accuracy%</td>
    <td>Drop</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MNLI</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>[m, mm] = [84.57, 84.79]</td>
    <td>[m, mm] = [82.45, 83.27]</td>
    <td>[-2.51, -1.80]</td>
    <td>[m, mm] = [83.20, 84.11]</td>
    <td>[-1.62, -0.80]</td>
  </tr>
  <tr>
    <td>SST-2</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>accuracy = 92.32</td>
    <td>accuracy = 91.51</td>
    <td>-0.88</td>
    <td>accuracy = 92.20</td>
    <td>-0.13</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>[accuracy, f1] = [91.10, 88.05]</td>
    <td>[accuracy, f1] = [90.48, 87.06]</td>
    <td>[-0.68, -1.12]</td>
    <td>[accuracy, f1] = [90.92, 87.78]</td>
    <td>[-0.20, -0.31]</td>
  </tr>
  <tr>
    <td>QNLI</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>accuracy = 91.54</td>
    <td>accuracy = 90.39</td>
    <td>-1.26</td>
    <td>accuracy = 90.87</td>
    <td>-0.73</td>
  </tr>
  <tr>
    <td>QnA</td>
    <td>PyTorch</td>
    <td>BERT base</td>
    <td>[em, f1] = [79.34, 87.10]</td>
    <td>[em, f1] = [77.27, 85.75]</td>
    <td>[-2.61, -1.54]</td>
    <td>[em, f1] = [78.03, 86.50]</td>
    <td>[-1.65, -0.69]</td>
  </tr>
</tbody>
</table>

<table class="docutils">
<thead>
  <tr>
    <th>Framework</th>
    <th>Model</th>
    <th>FP32 Baseline</th>
    <th>Compression</th>
    <th>Dataset</th>
    <th>Accuracy% (Drop)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>PyTorch</td>
    <td>ResNet18</td>
    <td>69.76</td>
    <td>30% Sparsity on Magnitude</td>
    <td>ImageNet</td>
    <td>69.47(-0.42)</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>ResNet18</td>
    <td>69.76</td>
    <td>30% Sparsity on Gradient Sensitivity</td>
    <td>ImageNet</td>
    <td>68.85(-1.30)</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>ResNet50</td>
    <td>76.13</td>
    <td>30% Sparsity on Magnitude</td>
    <td>ImageNet</td>
    <td>76.11(-0.03)</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>ResNet50</td>
    <td>76.13</td>
    <td>30% Sparsity on Magnitude and Post Training Quantization</td>
    <td>ImageNet</td>
    <td>76.01(-0.16)</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>ResNet50</td>
    <td>76.13</td>
    <td>30% Sparsity on Magnitude and Quantization Aware Training</td>
    <td>ImageNet</td>
    <td>75.90(-0.30)</td>
  </tr>
</tbody>
</table>

## Validated Knowledge Distillation Examples
<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Example Name</th>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Student<br>(Accuracy)</th>
    <th rowspan="2">Teacher<br>(Accuracy)</th>
    <th rowspan="2">Student With Distillation<br>(Accuracy Improvement)</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">ResNet example</td>
    <td rowspan="2">ImageNet</td>
    <td rowspan="2">ResNet18<br>(0.6739)</td>
    <td rowspan="2">ResNet50<br>(0.7399)</td>
    <td rowspan="2">0.6845<br>(0.0106)</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="2">BlendCNN example</td>
    <td rowspan="2">MRPC</td>
    <td rowspan="2">BlendCNN<br>(0.7034)</td>
    <td rowspan="2">BERT-Base<br>(0.8382)</td>
    <td rowspan="2">0.7034<br>(0)</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td rowspan="2">BiLSTM example</td>
    <td rowspan="2">SST-2</td>
    <td rowspan="2">BiLSTM<br>(0.7913)</td>
    <td rowspan="2">RoBERTa-Base<br>(0.9404)</td>
    <td rowspan="2">0.8085<br>(0.0172)</td>
  </tr>
  <tr>
  </tr>
</tbody>
</table>



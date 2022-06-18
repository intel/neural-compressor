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
    <td>WIP</td>
    <td></td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>Yes</td>
    <td><a href="../examples/pytorch/image_recognition/3d-unet/quantization">Link</a></td>
  </tr>
</tbody>
</table>

## Validated Quantization Examples

Performance results test on ​​06/07/2022 with Intel Xeon Platinum 8380 Scalable processor, using 1 socket, 4 cores/instance, 10 instances and batch size 1. 

Performance varies by use, configuration and other factors. See [platform configuration](./platform_configuration.md) for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks

### TensorFlow models with Intel TensorFlow 2.9.1

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
    <th rowspan="2">Example</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-za14">BERT large SQuAD</td>
    <td class="tg-za14">92.39</td>
    <td class="tg-za14">92.99</td>
    <td class="tg-za14">-0.64%</td>
    <td class="tg-za14">25.32</td>
    <td class="tg-za14">12.53</td>
    <td class="tg-za14">2.02x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-za14">DenseNet121</td>
    <td class="tg-za14">73.57%</td>
    <td class="tg-za14">72.89%</td>
    <td class="tg-za14">0.93%</td>
    <td class="tg-za14">370.52</td>
    <td class="tg-za14">329.74</td>
    <td class="tg-za14">1.12x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-za14">DenseNet161</td>
    <td class="tg-za14">76.24%</td>
    <td class="tg-za14">76.29%</td>
    <td class="tg-za14">-0.07%</td>
    <td class="tg-za14">219.46</td>
    <td class="tg-za14">180.75</td>
    <td class="tg-za14">1.21x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-6oje">DenseNet169</td>
    <td class="tg-za14">74.40%</td>
    <td class="tg-za14">74.65%</td>
    <td class="tg-za14">-0.33%</td>
    <td class="tg-za14">301.33</td>
    <td class="tg-za14">259.88</td>
    <td class="tg-za14">1.16x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-za14">Faster R-CNN Inception ResNet V2</td>
    <td class="tg-za14">37.98%</td>
    <td class="tg-za14">38.33%</td>
    <td class="tg-za14">-0.91%</td>
    <td class="tg-za14">3.96</td>
    <td class="tg-za14">2.34</td>
    <td class="tg-za14">1.69x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-za14">Faster R-CNN Inception ResNet V2 </td>
    <td class="tg-za14">37.84%</td>
    <td class="tg-za14">38.33%</td>
    <td class="tg-za14">-1.28%</td>
    <td class="tg-za14">3.98</td>
    <td class="tg-za14">2.31</td>
    <td class="tg-za14">1.72x</td>
    <td class="tg-7zrl">SavedModel</td>
  </tr>
  <tr>
    <td class="tg-za14">Faster R-CNN ResNet101</td>
    <td class="tg-za14">30.28%</td>
    <td class="tg-za14">30.39%</td>
    <td class="tg-za14">-0.36%</td>
    <td class="tg-za14">70</td>
    <td class="tg-za14">19.98</td>
    <td class="tg-za14">3.50x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN ResNet101</td>
    <td class="tg-7zrl">30.37%</td>
    <td class="tg-7zrl">30.39%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">70.26</td>
    <td class="tg-7zrl">16.98</td>
    <td class="tg-7zrl">4.14x</td>
    <td class="tg-7zrl">SavedModel</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception ResNet V2</td>
    <td class="tg-7zrl">80.44%</td>
    <td class="tg-7zrl">80.40%</td>
    <td class="tg-7zrl">0.05%</td>
    <td class="tg-7zrl">281.79</td>
    <td class="tg-7zrl">137.91</td>
    <td class="tg-7zrl">2.04x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception V1</td>
    <td class="tg-7zrl">70.48%</td>
    <td class="tg-7zrl">69.74%</td>
    <td class="tg-7zrl">1.06%</td>
    <td class="tg-7zrl">2193.17</td>
    <td class="tg-7zrl">975.6</td>
    <td class="tg-7zrl">2.25x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception V2</td>
    <td class="tg-7zrl">74.36%</td>
    <td class="tg-7zrl">73.97%</td>
    <td class="tg-7zrl">0.53%</td>
    <td class="tg-7zrl">1835.35</td>
    <td class="tg-7zrl">838.82</td>
    <td class="tg-7zrl">2.19x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception V3</td>
    <td class="tg-7zrl">77.28%</td>
    <td class="tg-7zrl">76.75%</td>
    <td class="tg-7zrl">0.69%</td>
    <td class="tg-7zrl">973.42</td>
    <td class="tg-7zrl">376.3</td>
    <td class="tg-7zrl">2.59x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception V4</td>
    <td class="tg-7zrl">80.40%</td>
    <td class="tg-7zrl">80.27%</td>
    <td class="tg-7zrl">0.16%</td>
    <td class="tg-7zrl">575.9</td>
    <td class="tg-7zrl">200.55</td>
    <td class="tg-7zrl">2.87x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Mask R-CNN Inception V2</td>
    <td class="tg-7zrl">28.53%</td>
    <td class="tg-7zrl">28.73%</td>
    <td class="tg-7zrl">-0.70%</td>
    <td class="tg-7zrl">132.51</td>
    <td class="tg-7zrl">50.3</td>
    <td class="tg-7zrl">2.63x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">Mask R-CNN Inception V2 </td>
    <td class="tg-7zrl">28.53%</td>
    <td class="tg-7zrl">28.73%</td>
    <td class="tg-7zrl">-0.70%</td>
    <td class="tg-7zrl">132.89</td>
    <td class="tg-7zrl">50.97</td>
    <td class="tg-7zrl">2.61x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MobileNet V1</td>
    <td class="tg-7zrl">71.79%</td>
    <td class="tg-7zrl">70.96%</td>
    <td class="tg-7zrl">1.17%</td>
    <td class="tg-7zrl">3545.79</td>
    <td class="tg-7zrl">1191.94</td>
    <td class="tg-7zrl">2.97x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">MobileNet V2</td>
    <td class="tg-7zrl">71.89%</td>
    <td class="tg-7zrl">71.76%</td>
    <td class="tg-7zrl">0.18%</td>
    <td class="tg-7zrl">2431.66</td>
    <td class="tg-7zrl">1420.11</td>
    <td class="tg-7zrl">1.71x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet101</td>
    <td class="tg-7zrl">77.50%</td>
    <td class="tg-7zrl">76.45%</td>
    <td class="tg-7zrl">1.37%</td>
    <td class="tg-7zrl">877.91</td>
    <td class="tg-7zrl">355.49</td>
    <td class="tg-7zrl">2.47x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNet50 Fashion</td>
    <td class="tg-7zrl">77.80%</td>
    <td class="tg-7zrl">78.12%</td>
    <td class="tg-7zrl">-0.41%</td>
    <td class="tg-7zrl">3977.5</td>
    <td class="tg-7zrl">2150.68</td>
    <td class="tg-7zrl">1.85x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50 V1.0</td>
    <td class="tg-7zrl">74.11%</td>
    <td class="tg-7zrl">74.27%</td>
    <td class="tg-7zrl">-0.22%</td>
    <td class="tg-7zrl">1509.64</td>
    <td class="tg-7zrl">472.66</td>
    <td class="tg-7zrl">3.19x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNet50 V1.5</td>
    <td class="tg-7zrl">76.82%</td>
    <td class="tg-7zrl">76.46%</td>
    <td class="tg-7zrl">0.47%</td>
    <td class="tg-7zrl">1260.01</td>
    <td class="tg-7zrl">415.83</td>
    <td class="tg-7zrl">3.03x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet V2 101</td>
    <td class="tg-7zrl">72.67%</td>
    <td class="tg-7zrl">71.87%</td>
    <td class="tg-7zrl">1.11%</td>
    <td class="tg-7zrl">436.52</td>
    <td class="tg-7zrl">318.3</td>
    <td class="tg-7zrl">1.37x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet V2 152</td>
    <td class="tg-7zrl">73.03%</td>
    <td class="tg-7zrl">72.37%</td>
    <td class="tg-7zrl">0.91%</td>
    <td class="tg-7zrl">306.82</td>
    <td class="tg-7zrl">221.4</td>
    <td class="tg-7zrl">1.39x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet V2 50</td>
    <td class="tg-7zrl">70.33%</td>
    <td class="tg-7zrl">69.64%</td>
    <td class="tg-7zrl">0.99%</td>
    <td class="tg-7zrl">749.85</td>
    <td class="tg-7zrl">574.19</td>
    <td class="tg-7zrl">1.31x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">SSD MobileNet V1</td>
    <td class="tg-7zrl">22.97%</td>
    <td class="tg-7zrl">23.13%</td>
    <td class="tg-7zrl">-0.69%</td>
    <td class="tg-7zrl">952.9</td>
    <td class="tg-7zrl">582.87</td>
    <td class="tg-7zrl">1.63x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">SSD MobileNet V1</td>
    <td class="tg-7zrl">22.99%</td>
    <td class="tg-7zrl">23.13%</td>
    <td class="tg-7zrl">-0.61%</td>
    <td class="tg-7zrl">954.92</td>
    <td class="tg-7zrl">413.24</td>
    <td class="tg-7zrl">2.31x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-zk71">SSD ResNet34</td>
    <td class="tg-7zrl">21.69%</td>
    <td class="tg-7zrl">22.09%</td>
    <td class="tg-7zrl">-1.81%</td>
    <td class="tg-7zrl">44.46</td>
    <td class="tg-7zrl">11.81</td>
    <td class="tg-7zrl">3.76x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">SSD ResNet50 V1</td>
    <td class="tg-7zrl">37.86%</td>
    <td class="tg-7zrl">38.00%</td>
    <td class="tg-7zrl">-0.37%</td>
    <td class="tg-7zrl">69.5</td>
    <td class="tg-7zrl">26.04</td>
    <td class="tg-7zrl">2.67x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">SSD ResNet50 V1</td>
    <td class="tg-7zrl">37.81%</td>
    <td class="tg-7zrl">38.00%</td>
    <td class="tg-7zrl">-0.50%</td>
    <td class="tg-7zrl">69.27</td>
    <td class="tg-7zrl">21.17</td>
    <td class="tg-7zrl">3.27x</td>
    <td class="tg-7zrl">ckpt</td>
  </tr>
  <tr>
    <td class="tg-zk71">VGG16</td>
    <td class="tg-7zrl">72.66%</td>
    <td class="tg-7zrl">70.89%</td>
    <td class="tg-7zrl">2.50%</td>
    <td class="tg-7zrl">660.46</td>
    <td class="tg-7zrl">177.85</td>
    <td class="tg-7zrl">3.71x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">VGG19</td>
    <td class="tg-7zrl">72.72%</td>
    <td class="tg-7zrl">71.01%</td>
    <td class="tg-7zrl">2.41%</td>
    <td class="tg-7zrl">562.04</td>
    <td class="tg-7zrl">147.61</td>
    <td class="tg-7zrl">3.81x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
  <tr>
    <td class="tg-zk71">Wide &amp; Deep</td>
    <td class="tg-7zrl">77.62%</td>
    <td class="tg-7zrl">77.67%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">21332.47</td>
    <td class="tg-7zrl">19714.08</td>
    <td class="tg-7zrl">1.08x</td>
    <td class="tg-7zrl">pb</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.11.0+cpu in PTQ mode

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
    <th rowspan="2">Example</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-iz6e">ALBERT base MRPC</td>
    <td class="tg-za14">88.06%</td>
    <td class="tg-za14">88.50%</td>
    <td class="tg-za14">-0.50%</td>
    <td class="tg-za14">34.28</td>
    <td class="tg-za14">29.54</td>
    <td class="tg-za14">1.16x</td>
    <td class="tg-za14">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">Barthez MRPC</td>
    <td class="tg-7zrl">82.99%</td>
    <td class="tg-7zrl">83.81%</td>
    <td class="tg-7zrl">-0.97%</td>
    <td class="tg-7zrl">166.84</td>
    <td class="tg-7zrl">89.56</td>
    <td class="tg-7zrl">1.86x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base COLA</td>
    <td class="tg-7zrl">58.80%</td>
    <td class="tg-7zrl">58.84%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">260</td>
    <td class="tg-7zrl">126.47</td>
    <td class="tg-7zrl">2.06x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-zk71">BERT base MRPC</td>
    <td class="tg-7zrl">90.28%</td>
    <td class="tg-7zrl">90.69%</td>
    <td class="tg-7zrl">-0.45%</td>
    <td class="tg-7zrl">251.79</td>
    <td class="tg-7zrl">126.46</td>
    <td class="tg-7zrl">1.99x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base RTE</td>
    <td class="tg-7zrl">69.31%</td>
    <td class="tg-7zrl">69.68%</td>
    <td class="tg-7zrl">-0.52%</td>
    <td class="tg-7zrl">252.14</td>
    <td class="tg-7zrl">126.45</td>
    <td class="tg-7zrl">1.99x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base SST2</td>
    <td class="tg-7zrl">91.97%</td>
    <td class="tg-7zrl">91.86%</td>
    <td class="tg-7zrl">0.12%</td>
    <td class="tg-7zrl">258.98</td>
    <td class="tg-7zrl">126.42</td>
    <td class="tg-7zrl">2.05x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base STSB</td>
    <td class="tg-7zrl">89.13%</td>
    <td class="tg-7zrl">89.75%</td>
    <td class="tg-7zrl">-0.68%</td>
    <td class="tg-7zrl">249.57</td>
    <td class="tg-7zrl">126.39</td>
    <td class="tg-7zrl">1.97x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT large COLA</td>
    <td class="tg-7zrl">62.88%</td>
    <td class="tg-7zrl">62.57%</td>
    <td class="tg-7zrl">0.49%</td>
    <td class="tg-7zrl">88.75</td>
    <td class="tg-7zrl">36.7</td>
    <td class="tg-7zrl">2.42x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT large MRPC</td>
    <td class="tg-7zrl">89.93%</td>
    <td class="tg-7zrl">90.38%</td>
    <td class="tg-7zrl">-0.49%</td>
    <td class="tg-7zrl">89.43</td>
    <td class="tg-7zrl">36.62</td>
    <td class="tg-7zrl">2.44x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT large QNLI</td>
    <td class="tg-7zrl">90.96%</td>
    <td class="tg-7zrl">91.82%</td>
    <td class="tg-7zrl">-0.94%</td>
    <td class="tg-7zrl">91.27</td>
    <td class="tg-7zrl">37</td>
    <td class="tg-7zrl">2.47x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT large RTE</td>
    <td class="tg-7zrl">71.84%</td>
    <td class="tg-7zrl">72.56%</td>
    <td class="tg-7zrl">-1.00%</td>
    <td class="tg-7zrl">77.62</td>
    <td class="tg-7zrl">36.01</td>
    <td class="tg-7zrl">2.16x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">CamemBERT base MRPC</td>
    <td class="tg-7zrl">86.56%</td>
    <td class="tg-7zrl">86.82%</td>
    <td class="tg-7zrl">-0.30%</td>
    <td class="tg-7zrl">241.39</td>
    <td class="tg-7zrl">124.77</td>
    <td class="tg-7zrl">1.93x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Deberta MRPC</td>
    <td class="tg-7zrl">91.17%</td>
    <td class="tg-7zrl">90.91%</td>
    <td class="tg-7zrl">0.28%</td>
    <td class="tg-7zrl">152.09</td>
    <td class="tg-7zrl">85.13</td>
    <td class="tg-7zrl">1.79x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">DistilBERT base MRPC</td>
    <td class="tg-7zrl">88.66%</td>
    <td class="tg-7zrl">89.16%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">415.09</td>
    <td class="tg-7zrl">246.9</td>
    <td class="tg-7zrl">1.68x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">DistilBERT base MRPC</td>
    <td class="tg-7zrl">88.74%</td>
    <td class="tg-7zrl">89.16%</td>
    <td class="tg-7zrl">-0.47%</td>
    <td class="tg-7zrl">459.93</td>
    <td class="tg-7zrl">245.33</td>
    <td class="tg-7zrl">1.87x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">FlauBERT MRPC</td>
    <td class="tg-7zrl">81.01%</td>
    <td class="tg-7zrl">80.19%</td>
    <td class="tg-7zrl">1.01%</td>
    <td class="tg-7zrl">644.05</td>
    <td class="tg-7zrl">457.32</td>
    <td class="tg-7zrl">1.41x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">Inception V3</td>
    <td class="tg-7zrl">69.43%</td>
    <td class="tg-7zrl">69.52%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">454.3</td>
    <td class="tg-7zrl">213.7</td>
    <td class="tg-7zrl">2.13x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Longformer MRPC</td>
    <td class="tg-7zrl">90.59%</td>
    <td class="tg-7zrl">91.46%</td>
    <td class="tg-7zrl">-0.95%</td>
    <td class="tg-7zrl">21.51</td>
    <td class="tg-7zrl">17.45</td>
    <td class="tg-7zrl">1.23x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">Mask R-CNN</td>
    <td class="tg-7zrl">37.70%</td>
    <td class="tg-7zrl">37.80%</td>
    <td class="tg-7zrl">-0.26%</td>
    <td class="tg-7zrl">17.61</td>
    <td class="tg-7zrl">5.76</td>
    <td class="tg-7zrl">3.06x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">mBart WNLI</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">65.05</td>
    <td class="tg-7zrl">31.26</td>
    <td class="tg-7zrl">2.08x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileNet V2</td>
    <td class="tg-7zrl">70.54%</td>
    <td class="tg-7zrl">71.84%</td>
    <td class="tg-7zrl">-1.81%</td>
    <td class="tg-7zrl">740.97</td>
    <td class="tg-7zrl">535.54</td>
    <td class="tg-7zrl">1.38x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">lvwerra/pegasus-samsum</td>
    <td class="tg-7zrl">42.21</td>
    <td class="tg-7zrl">42.67</td>
    <td class="tg-7zrl">-1.09%</td>
    <td class="tg-7zrl">3.89</td>
    <td class="tg-7zrl">1.14</td>
    <td class="tg-7zrl">3.41x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">PeleeNet</td>
    <td class="tg-7zrl">71.64%</td>
    <td class="tg-7zrl">72.10%</td>
    <td class="tg-7zrl">-0.64%</td>
    <td class="tg-7zrl">502.01</td>
    <td class="tg-7zrl">391.31</td>
    <td class="tg-7zrl">1.28x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet18 </td>
    <td class="tg-7zrl">69.57%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.27%</td>
    <td class="tg-7zrl">800.43</td>
    <td class="tg-7zrl">381.27</td>
    <td class="tg-7zrl">2.10x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet18 </td>
    <td class="tg-7zrl">69.57%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.28%</td>
    <td class="tg-7zrl">811.09</td>
    <td class="tg-7zrl">389.36</td>
    <td class="tg-7zrl">2.08x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50</td>
    <td class="tg-7zrl">75.98%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">-0.21%</td>
    <td class="tg-7zrl">507.55</td>
    <td class="tg-7zrl">200.52</td>
    <td class="tg-7zrl">2.53x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNeXt101_32x8d</td>
    <td class="tg-7zrl">79.08%</td>
    <td class="tg-7zrl">79.31%</td>
    <td class="tg-7zrl">-0.29%</td>
    <td class="tg-7zrl">203.54</td>
    <td class="tg-7zrl">73.85</td>
    <td class="tg-7zrl">2.76x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">RNN-T</td>
    <td class="tg-7zrl">92.45</td>
    <td class="tg-7zrl">92.55</td>
    <td class="tg-7zrl">-0.10%</td>
    <td class="tg-7zrl">79.21</td>
    <td class="tg-7zrl">20.47</td>
    <td class="tg-7zrl">3.87x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">Roberta Base MRPC</td>
    <td class="tg-7zrl">87.88%</td>
    <td class="tg-7zrl">88.18%</td>
    <td class="tg-7zrl">-0.34%</td>
    <td class="tg-7zrl">250.21</td>
    <td class="tg-7zrl">124.92</td>
    <td class="tg-7zrl">2.00x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">Se_ResNeXt50_32x4d</td>
    <td class="tg-7zrl">78.98%</td>
    <td class="tg-7zrl">79.08%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">358.63</td>
    <td class="tg-7zrl">173.03</td>
    <td class="tg-7zrl">2.07x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SqueezeBERT MRPC</td>
    <td class="tg-7zrl">87.77%</td>
    <td class="tg-7zrl">87.65%</td>
    <td class="tg-7zrl">0.14%</td>
    <td class="tg-7zrl">249.89</td>
    <td class="tg-7zrl">207.43</td>
    <td class="tg-7zrl">1.20x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Transfo-xl MRPC</td>
    <td class="tg-7zrl">81.97%</td>
    <td class="tg-7zrl">81.20%</td>
    <td class="tg-7zrl">0.94%</td>
    <td class="tg-7zrl">11.25</td>
    <td class="tg-7zrl">8.34</td>
    <td class="tg-7zrl">1.35x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">YOLOv3</td>
    <td class="tg-7zrl">24.60%</td>
    <td class="tg-7zrl">24.54%</td>
    <td class="tg-7zrl">0.21%</td>
    <td class="tg-7zrl">108.09</td>
    <td class="tg-7zrl">40.02</td>
    <td class="tg-7zrl">2.70x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
</tbody>
</table>

### PyTorch models with Torch 1.11.0+cpu in QAT mode
<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
    <th rowspan="2">Example</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-iz6e">ResNet18</td>
    <td class="tg-za14">69.74%</td>
    <td class="tg-za14">69.76%</td>
    <td class="tg-za14">-0.03%</td>
    <td class="tg-za14">804.76</td>
    <td class="tg-za14">388.67</td>
    <td class="tg-za14">2.07x</td>
    <td class="tg-za14">eager</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet18</td>
    <td class="tg-7zrl">69.73%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.04%</td>
    <td class="tg-7zrl">806.44</td>
    <td class="tg-7zrl">386.59</td>
    <td class="tg-7zrl">2.09x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base MRPC QAT</td>
    <td class="tg-7zrl">89.60%</td>
    <td class="tg-7zrl">89.50%</td>
    <td class="tg-7zrl">0.11%</td>
    <td class="tg-7zrl">258.89</td>
    <td class="tg-7zrl">125.79</td>
    <td class="tg-7zrl">2.06x</td>
    <td class="tg-7zrl">fx</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50</td>
    <td class="tg-7zrl">76.04%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">-0.14%</td>
    <td class="tg-7zrl">490.64</td>
    <td class="tg-7zrl">203.49</td>
    <td class="tg-7zrl">2.41x</td>
    <td class="tg-7zrl">eager</td>
  </tr>
</tbody>
</table>

### PyTorch models with IPEX 1.11.0

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
    <th rowspan="2">Example</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-za14">bert-large-uncased-whole-word-masking-finetuned-squad </td>
    <td class="tg-za14">92.9</td>
    <td class="tg-za14">93.16</td>
    <td class="tg-za14">-0.28%</td>
    <td class="tg-za14">37.13</td>
    <td class="tg-za14">11.45</td>
    <td class="tg-za14">3.24x</td>
    <td class="tg-za14">ipex</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNeXt101_32x16d_wsl</td>
    <td class="tg-7zrl">84.02%</td>
    <td class="tg-7zrl">84.17%</td>
    <td class="tg-7zrl">-0.18%</td>
    <td class="tg-7zrl">163.45</td>
    <td class="tg-7zrl">28.9</td>
    <td class="tg-7zrl">5.66x</td>
    <td class="tg-7zrl">ipex</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNet50</td>
    <td class="tg-7zrl">76.00%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">707.86</td>
    <td class="tg-7zrl">202.02</td>
    <td class="tg-7zrl">3.51x</td>
    <td class="tg-7zrl">ipex</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD ResNet34</td>
    <td class="tg-7zrl">19.97%</td>
    <td class="tg-7zrl">20.00%</td>
    <td class="tg-7zrl">-0.15%</td>
    <td class="tg-7zrl">30.84</td>
    <td class="tg-7zrl">8.55</td>
    <td class="tg-7zrl">3.61x</td>
    <td class="tg-7zrl">ipex</td>
  </tr>
</tbody>
</table>

### ONNX Models with ONNX Runtime 1.11.0

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
    <th rowspan="2">Example</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-iz6e">AlexNet</td>
    <td class="tg-za14">54.74%</td>
    <td class="tg-za14">54.79%</td>
    <td class="tg-za14">-0.09%</td>
    <td class="tg-za14">1518.97</td>
    <td class="tg-za14">676.74</td>
    <td class="tg-za14">2.24x</td>
    <td class="tg-za14">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">AlexNet </td>
    <td class="tg-7zrl">54.74%</td>
    <td class="tg-7zrl">54.79%</td>
    <td class="tg-7zrl">-0.09%</td>
    <td class="tg-7zrl">1411.3</td>
    <td class="tg-7zrl">652.6</td>
    <td class="tg-7zrl">2.16x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT base MRPC DYNAMIC</td>
    <td class="tg-7zrl">85.54%</td>
    <td class="tg-7zrl">86.03%</td>
    <td class="tg-7zrl">-0.57%</td>
    <td class="tg-7zrl">379.71</td>
    <td class="tg-7zrl">156.16</td>
    <td class="tg-7zrl">2.43x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">BERT base MRPC STATIC</td>
    <td class="tg-7zrl">85.29%</td>
    <td class="tg-7zrl">86.03%</td>
    <td class="tg-7zrl">-0.86%</td>
    <td class="tg-7zrl">756.33</td>
    <td class="tg-7zrl">316.36</td>
    <td class="tg-7zrl">2.39x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT SQuAD</td>
    <td class="tg-7zrl">80.44</td>
    <td class="tg-7zrl">80.67</td>
    <td class="tg-7zrl">-0.29%</td>
    <td class="tg-7zrl">115.58</td>
    <td class="tg-7zrl">64.71</td>
    <td class="tg-7zrl">1.79x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">BERT SQuAD</td>
    <td class="tg-7zrl">80.44</td>
    <td class="tg-7zrl">80.67</td>
    <td class="tg-7zrl">-0.29%</td>
    <td class="tg-7zrl">115.4</td>
    <td class="tg-7zrl">64.68</td>
    <td class="tg-7zrl">1.78x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">CaffeNet</td>
    <td class="tg-7zrl">56.19%</td>
    <td class="tg-7zrl">56.30%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">2786.79</td>
    <td class="tg-7zrl">802.7</td>
    <td class="tg-7zrl">3.47x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">CaffeNet</td>
    <td class="tg-7zrl">56.19%</td>
    <td class="tg-7zrl">56.30%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">2726.86</td>
    <td class="tg-7zrl">819.41</td>
    <td class="tg-7zrl">3.33x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">DenseNet</td>
    <td class="tg-7zrl">60.20%</td>
    <td class="tg-7zrl">60.96%</td>
    <td class="tg-7zrl">-1.25%</td>
    <td class="tg-7zrl">404.83</td>
    <td class="tg-7zrl">340.63</td>
    <td class="tg-7zrl">1.19x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">DistilBERT base MRPC</td>
    <td class="tg-7zrl">84.56%</td>
    <td class="tg-7zrl">84.56%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">1630.41</td>
    <td class="tg-7zrl">596.68</td>
    <td class="tg-7zrl">2.73x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">EfficientNet</td>
    <td class="tg-7zrl">77.58%</td>
    <td class="tg-7zrl">77.70%</td>
    <td class="tg-7zrl">-0.15%</td>
    <td class="tg-7zrl">1985.35</td>
    <td class="tg-7zrl">1097.33</td>
    <td class="tg-7zrl">1.81x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN</td>
    <td class="tg-7zrl">33.99%</td>
    <td class="tg-7zrl">34.37%</td>
    <td class="tg-7zrl">-1.11%</td>
    <td class="tg-7zrl">10.02</td>
    <td class="tg-7zrl">4.32</td>
    <td class="tg-7zrl">2.32x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Faster R-CNN</td>
    <td class="tg-7zrl">33.94%</td>
    <td class="tg-7zrl">34.37%</td>
    <td class="tg-7zrl">-1.25%</td>
    <td class="tg-7zrl">10.41</td>
    <td class="tg-7zrl">4.28</td>
    <td class="tg-7zrl">2.43x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-zk71">FCN</td>
    <td class="tg-7zrl">64.66%</td>
    <td class="tg-7zrl">64.98%</td>
    <td class="tg-7zrl">-0.49%</td>
    <td class="tg-7zrl">44.31</td>
    <td class="tg-7zrl">14.2</td>
    <td class="tg-7zrl">3.12x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">FCN</td>
    <td class="tg-7zrl">64.66%</td>
    <td class="tg-7zrl">64.98%</td>
    <td class="tg-7zrl">-0.49%</td>
    <td class="tg-7zrl">18.11</td>
    <td class="tg-7zrl">14.19</td>
    <td class="tg-7zrl">1.28x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">GoogleNet</td>
    <td class="tg-7zrl">67.61%</td>
    <td class="tg-7zrl">67.79%</td>
    <td class="tg-7zrl">-0.27%</td>
    <td class="tg-7zrl">1165.84</td>
    <td class="tg-7zrl">810.65</td>
    <td class="tg-7zrl">1.44x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">GoogleNet</td>
    <td class="tg-7zrl">67.61%</td>
    <td class="tg-7zrl">67.79%</td>
    <td class="tg-7zrl">-0.27%</td>
    <td class="tg-7zrl">1165.73</td>
    <td class="tg-7zrl">809.98</td>
    <td class="tg-7zrl">1.44x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Inception V1</td>
    <td class="tg-7zrl">67.23%</td>
    <td class="tg-7zrl">67.24%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1205.89</td>
    <td class="tg-7zrl">838.71</td>
    <td class="tg-7zrl">1.44x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">Inception V1</td>
    <td class="tg-7zrl">67.23%</td>
    <td class="tg-7zrl">67.24%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1204.93</td>
    <td class="tg-7zrl">843.16</td>
    <td class="tg-7zrl">1.43x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Mask R-CNN</td>
    <td class="tg-7zrl">33.40%</td>
    <td class="tg-7zrl">33.72%</td>
    <td class="tg-7zrl">-0.95%</td>
    <td class="tg-7zrl">8.56</td>
    <td class="tg-7zrl">3.76</td>
    <td class="tg-7zrl">2.27x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">Mask R-CNN</td>
    <td class="tg-7zrl">33.33%</td>
    <td class="tg-7zrl">33.72%</td>
    <td class="tg-7zrl">-1.16%</td>
    <td class="tg-7zrl">8.4</td>
    <td class="tg-7zrl">3.81</td>
    <td class="tg-7zrl">2.20x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-zk71">Mobile bert MRPC</td>
    <td class="tg-7zrl">86.03%</td>
    <td class="tg-7zrl">86.27%</td>
    <td class="tg-7zrl">-0.28%</td>
    <td class="tg-7zrl">790.11</td>
    <td class="tg-7zrl">686.35</td>
    <td class="tg-7zrl">1.15x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileBERT SQuAD MLPerf</td>
    <td class="tg-7zrl">89.84</td>
    <td class="tg-7zrl">90.03</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">102.92</td>
    <td class="tg-7zrl">95.19</td>
    <td class="tg-7zrl">1.08x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileNet V2</td>
    <td class="tg-7zrl">65.47%</td>
    <td class="tg-7zrl">66.89%</td>
    <td class="tg-7zrl">-2.12%</td>
    <td class="tg-7zrl">5133.84</td>
    <td class="tg-7zrl">3394.73</td>
    <td class="tg-7zrl">1.51x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileNet V2</td>
    <td class="tg-7zrl">65.47%</td>
    <td class="tg-7zrl">66.89%</td>
    <td class="tg-7zrl">-2.12%</td>
    <td class="tg-7zrl">5066.31</td>
    <td class="tg-7zrl">3386.3</td>
    <td class="tg-7zrl">1.50x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileNet V3 MLPerf</td>
    <td class="tg-7zrl">75.59%</td>
    <td class="tg-7zrl">75.74%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">4133.22</td>
    <td class="tg-7zrl">2132.92</td>
    <td class="tg-7zrl">1.94x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">MobileNetV2 (ONNX Model Zoo)</td>
    <td class="tg-7zrl">68.30%</td>
    <td class="tg-7zrl">69.48%</td>
    <td class="tg-7zrl">-1.70%</td>
    <td class="tg-7zrl">5349.42</td>
    <td class="tg-7zrl">3373.29</td>
    <td class="tg-7zrl">1.59x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50 V1.5 MLPerf</td>
    <td class="tg-7zrl">76.13%</td>
    <td class="tg-7zrl">76.46%</td>
    <td class="tg-7zrl">-0.43%</td>
    <td class="tg-7zrl">1139.56</td>
    <td class="tg-7zrl">549.88</td>
    <td class="tg-7zrl">2.07x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50 V1.5</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.29%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1165.35</td>
    <td class="tg-7zrl">556.02</td>
    <td class="tg-7zrl">2.10x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">ResNet50 V1.5</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.29%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1319.32</td>
    <td class="tg-7zrl">543.44</td>
    <td class="tg-7zrl">2.43x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td class="tg-7zrl">74.76%</td>
    <td class="tg-7zrl">74.99%</td>
    <td class="tg-7zrl">-0.31%</td>
    <td class="tg-7zrl">1363.39</td>
    <td class="tg-7zrl">573.1</td>
    <td class="tg-7zrl">2.38x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Roberta Base MRPC</td>
    <td class="tg-7zrl">90.44%</td>
    <td class="tg-7zrl">89.95%</td>
    <td class="tg-7zrl">0.54%</td>
    <td class="tg-7zrl">811.05</td>
    <td class="tg-7zrl">312.71</td>
    <td class="tg-7zrl">2.59x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-zk71">ShuffleNet V2</td>
    <td class="tg-7zrl">66.13%</td>
    <td class="tg-7zrl">66.36%</td>
    <td class="tg-7zrl">-0.35%</td>
    <td class="tg-7zrl">4948.77</td>
    <td class="tg-7zrl">2847.66</td>
    <td class="tg-7zrl">1.74x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SqueezeNet</td>
    <td class="tg-7zrl">56.55%</td>
    <td class="tg-7zrl">56.87%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">6296.79</td>
    <td class="tg-7zrl">4340.51</td>
    <td class="tg-7zrl">1.45x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SqueezeNet</td>
    <td class="tg-7zrl">56.55%</td>
    <td class="tg-7zrl">56.87%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">6227.76</td>
    <td class="tg-7zrl">4383.8</td>
    <td class="tg-7zrl">1.42x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V1</td>
    <td class="tg-7zrl">22.20%</td>
    <td class="tg-7zrl">23.10%</td>
    <td class="tg-7zrl">-3.90%</td>
    <td class="tg-7zrl">917.64</td>
    <td class="tg-7zrl">709.48</td>
    <td class="tg-7zrl">1.29x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V1</td>
    <td class="tg-7zrl">22.20%</td>
    <td class="tg-7zrl">23.10%</td>
    <td class="tg-7zrl">-3.90%</td>
    <td class="tg-7zrl">840.99</td>
    <td class="tg-7zrl">655.99</td>
    <td class="tg-7zrl">1.28x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td class="tg-7zrl">22.88%</td>
    <td class="tg-7zrl">23.03%</td>
    <td class="tg-7zrl">-0.65%</td>
    <td class="tg-7zrl">845.17</td>
    <td class="tg-7zrl">666.25</td>
    <td class="tg-7zrl">1.27x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td class="tg-7zrl">22.88%</td>
    <td class="tg-7zrl">23.03%</td>
    <td class="tg-7zrl">-0.65%</td>
    <td class="tg-7zrl">790.06</td>
    <td class="tg-7zrl">624.2</td>
    <td class="tg-7zrl">1.27x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD MobileNet V2</td>
    <td class="tg-7zrl">23.83%</td>
    <td class="tg-7zrl">24.68%</td>
    <td class="tg-7zrl">-3.44%</td>
    <td class="tg-7zrl">703.55</td>
    <td class="tg-7zrl">506.6</td>
    <td class="tg-7zrl">1.39x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">SSD</td>
    <td class="tg-7zrl">18.68%</td>
    <td class="tg-7zrl">18.98%</td>
    <td class="tg-7zrl">-1.58%</td>
    <td class="tg-7zrl">41.99</td>
    <td class="tg-7zrl">11.12</td>
    <td class="tg-7zrl">3.78x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
  <tr>
    <td class="tg-7zrl">Tiny YOLOv3</td>
    <td class="tg-7zrl">12.08%</td>
    <td class="tg-7zrl">12.43%</td>
    <td class="tg-7zrl">-2.82%</td>
    <td class="tg-7zrl">836.21</td>
    <td class="tg-7zrl">659.69</td>
    <td class="tg-7zrl">1.27x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">VGG16</td>
    <td class="tg-7zrl">66.60%</td>
    <td class="tg-7zrl">66.69%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">312.48</td>
    <td class="tg-7zrl">128.98</td>
    <td class="tg-7zrl">2.42x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">VGG16 (ONNX Model Zoo)</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.40%</td>
    <td class="tg-7zrl">-0.17%</td>
    <td class="tg-7zrl">446.13</td>
    <td class="tg-7zrl">131.04</td>
    <td class="tg-7zrl">3.40x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">YOLOv3</td>
    <td class="tg-7zrl">26.88%</td>
    <td class="tg-7zrl">28.74%</td>
    <td class="tg-7zrl">-6.47%</td>
    <td class="tg-7zrl">157.39</td>
    <td class="tg-7zrl">66.72</td>
    <td class="tg-7zrl">2.36x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">YOLOv4</td>
    <td class="tg-7zrl">33.18%</td>
    <td class="tg-7zrl">33.71%</td>
    <td class="tg-7zrl">-1.57%</td>
    <td class="tg-7zrl">58.55</td>
    <td class="tg-7zrl">38.09</td>
    <td class="tg-7zrl">1.54x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ZFNet</td>
    <td class="tg-7zrl">55.89%</td>
    <td class="tg-7zrl">55.96%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">664.37</td>
    <td class="tg-7zrl">358.62</td>
    <td class="tg-7zrl">1.85x</td>
    <td class="tg-7zrl">qlinearops</td>
  </tr>
  <tr>
    <td class="tg-7zrl">ZFNet</td>
    <td class="tg-7zrl">55.89%</td>
    <td class="tg-7zrl">55.96%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">666.99</td>
    <td class="tg-7zrl">354.38</td>
    <td class="tg-7zrl">1.88x</td>
    <td class="tg-7zrl">qdq</td>
  </tr>
</tbody>
</table>

### MXNet models with MXNet 1.7.0

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>throughput (samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-za14">Inception V3</td>
    <td class="tg-za14">77.80%</td>
    <td class="tg-za14">77.65%</td>
    <td class="tg-za14">0.20%</td>
    <td class="tg-za14">920.74</td>
    <td class="tg-za14">276.73</td>
    <td class="tg-za14">3.33x</td>
  </tr>
  <tr>
    <td class="tg-za14">MobileNet V1</td>
    <td class="tg-za14">71.60%</td>
    <td class="tg-za14">72.23%</td>
    <td class="tg-za14">-0.86%</td>
    <td class="tg-za14">6585.19</td>
    <td class="tg-za14">2529.21</td>
    <td class="tg-za14">2.60x</td>
  </tr>
  <tr>
    <td class="tg-za14">MobileNet V2</td>
    <td class="tg-za14">70.80%</td>
    <td class="tg-za14">70.87%</td>
    <td class="tg-za14">-0.10%</td>
    <td class="tg-za14">5230.32</td>
    <td class="tg-za14">1996.47</td>
    <td class="tg-za14">2.62x</td>
  </tr>
  <tr>
    <td class="tg-6oje">ResNet V1 152</td>
    <td class="tg-za14">78.28%</td>
    <td class="tg-za14">78.54%</td>
    <td class="tg-za14">-0.33%</td>
    <td class="tg-za14">574.85</td>
    <td class="tg-za14">156.2</td>
    <td class="tg-za14">3.68x</td>
  </tr>
  <tr>
    <td class="tg-za14">ResNet50 V1.0</td>
    <td class="tg-za14">75.91%</td>
    <td class="tg-za14">76.33%</td>
    <td class="tg-za14">-0.55%</td>
    <td class="tg-za14">1567.9</td>
    <td class="tg-za14">427.99</td>
    <td class="tg-za14">3.66x</td>
  </tr>
  <tr>
    <td class="tg-za14">SqueezeNet</td>
    <td class="tg-za14">56.80%</td>
    <td class="tg-za14">56.97%</td>
    <td class="tg-za14">-0.28%</td>
    <td class="tg-za14">4704.51</td>
    <td class="tg-za14">1332.29</td>
    <td class="tg-za14">3.53x</td>
  </tr>
  <tr>
    <td class="tg-za14">SSD MobileNet V1</td>
    <td class="tg-za14">74.94%</td>
    <td class="tg-za14">75.54%</td>
    <td class="tg-za14">-0.79%</td>
    <td class="tg-za14">769.26</td>
    <td class="tg-za14">193.03</td>
    <td class="tg-za14">3.99x</td>
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



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
    <td rowspan="2">ResNet50 v1.5</td>
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
    <td rowspan="2">BERT-large</td>
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
    <td rowspan="2">SSD-ResNet34</td>
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

## Full Validated Models on Intel Xeon Platinum 8380 Scalable processor

The below tables are models enabled by the Intel® Neural Compressor. 

Performance varies by use, configuration and other factors. See backup for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks

Performance results are based on testing as of ​​04/08/2022 and may not reflect all publicly available ​updates. No product or component can be absolutely secure.

Intel optimizations, for Intel compilers or other products, may not optimize to the same degree for non-Intel products.

Your costs may vary.

Intel technologies may require enabled hardware, software or service activation.

© Intel Corporation.  Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others​.​​

### TensorFlow 2.x models

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnet50v1.5</td>
    <td class="tg-7zrl">76.82%</td>
    <td class="tg-7zrl">76.46%</td>
    <td class="tg-7zrl">0.47%</td>
    <td class="tg-7zrl">1239.52</td>
    <td class="tg-7zrl">433.07</td>
    <td class="tg-7zrl">2.86x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnet101</td>
    <td class="tg-7zrl">77.50%</td>
    <td class="tg-7zrl">76.45%</td>
    <td class="tg-7zrl">1.37%</td>
    <td class="tg-7zrl">874.41</td>
    <td class="tg-7zrl">352.91</td>
    <td class="tg-7zrl">2.48x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">inception_v2</td>
    <td class="tg-7zrl">74.36%</td>
    <td class="tg-7zrl">73.97%</td>
    <td class="tg-7zrl">0.53%</td>
    <td class="tg-7zrl">1840.78</td>
    <td class="tg-7zrl">853.52</td>
    <td class="tg-7zrl">2.16x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">inception_v3</td>
    <td class="tg-7zrl">77.28%</td>
    <td class="tg-7zrl">76.75%</td>
    <td class="tg-7zrl">0.69%</td>
    <td class="tg-7zrl">954.63</td>
    <td class="tg-7zrl">391.35</td>
    <td class="tg-7zrl">2.44x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">inception_v4</td>
    <td class="tg-7zrl">80.40%</td>
    <td class="tg-7zrl">80.27%</td>
    <td class="tg-7zrl">0.16%</td>
    <td class="tg-7zrl">580.02</td>
    <td class="tg-7zrl">202.14</td>
    <td class="tg-7zrl">2.87x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">mobilenetv1</td>
    <td class="tg-7zrl">71.79%</td>
    <td class="tg-7zrl">70.96%</td>
    <td class="tg-7zrl">1.17%</td>
    <td class="tg-7zrl">3587.79</td>
    <td class="tg-7zrl">1343.07</td>
    <td class="tg-7zrl">2.67x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">mobilenetv2</td>
    <td class="tg-7zrl">71.89%</td>
    <td class="tg-7zrl">71.76%</td>
    <td class="tg-7zrl">0.18%</td>
    <td class="tg-7zrl">2469.92</td>
    <td class="tg-7zrl">1434.87</td>
    <td class="tg-7zrl">1.72x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">ssd_resnet50_v1</td>
    <td class="tg-7zrl">37.86%</td>
    <td class="tg-7zrl">38.00%</td>
    <td class="tg-7zrl">-0.37%</td>
    <td class="tg-7zrl">70.35</td>
    <td class="tg-7zrl">26.34</td>
    <td class="tg-7zrl">2.67x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v1</td>
    <td class="tg-7zrl">22.97%</td>
    <td class="tg-7zrl">23.13%</td>
    <td class="tg-7zrl">-0.69%</td>
    <td class="tg-7zrl">852.80</td>
    <td class="tg-7zrl">460.33</td>
    <td class="tg-7zrl">1.85x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">faster_rcnn_inception_resnet_v2</td>
    <td class="tg-7zrl">37.99%</td>
    <td class="tg-7zrl">38.33%</td>
    <td class="tg-7zrl">-0.89%</td>
    <td class="tg-7zrl">4.06</td>
    <td class="tg-7zrl">2.33</td>
    <td class="tg-7zrl">1.74x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">faster_rcnn_resnet101_saved</td>
    <td class="tg-7zrl">30.37%</td>
    <td class="tg-7zrl">30.39%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">69.69</td>
    <td class="tg-7zrl">17.71</td>
    <td class="tg-7zrl">3.94x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">mask_rcnn_inception_v2</td>
    <td class="tg-7zrl">28.54%</td>
    <td class="tg-7zrl">28.72%</td>
    <td class="tg-7zrl">-0.63%</td>
    <td class="tg-7zrl">123.97</td>
    <td class="tg-7zrl">53.23</td>
    <td class="tg-7zrl">2.33x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">wide_deep_large_ds</td>
    <td class="tg-7zrl">77.62%</td>
    <td class="tg-7zrl">77.67%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">22704.16</td>
    <td class="tg-7zrl">21249.52</td>
    <td class="tg-7zrl">1.07x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">vgg16</td>
    <td class="tg-7zrl">72.66%</td>
    <td class="tg-7zrl">70.89%</td>
    <td class="tg-7zrl">2.50%</td>
    <td class="tg-7zrl">669.62</td>
    <td class="tg-7zrl">178.75</td>
    <td class="tg-7zrl">3.75x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">vgg19</td>
    <td class="tg-7zrl">72.72%</td>
    <td class="tg-7zrl">71.01%</td>
    <td class="tg-7zrl">2.41%</td>
    <td class="tg-7zrl">558.43</td>
    <td class="tg-7zrl">148.19</td>
    <td class="tg-7zrl">3.77x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnetv2_50</td>
    <td class="tg-7zrl">70.33%</td>
    <td class="tg-7zrl">69.64%</td>
    <td class="tg-7zrl">0.99%</td>
    <td class="tg-7zrl">765.73</td>
    <td class="tg-7zrl">580.54</td>
    <td class="tg-7zrl">1.32x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">densenet121</td>
    <td class="tg-7zrl">73.57%</td>
    <td class="tg-7zrl">72.89%</td>
    <td class="tg-7zrl">0.93%</td>
    <td class="tg-7zrl">366.59</td>
    <td class="tg-7zrl">296.63</td>
    <td class="tg-7zrl">1.24x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">densenet161</td>
    <td class="tg-7zrl">76.24%</td>
    <td class="tg-7zrl">76.29%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">218.26</td>
    <td class="tg-7zrl">164.48</td>
    <td class="tg-7zrl">1.33x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">densenet169</td>
    <td class="tg-7zrl">74.40%</td>
    <td class="tg-7zrl">74.65%</td>
    <td class="tg-7zrl">-0.33%</td>
    <td class="tg-7zrl">294.82</td>
    <td class="tg-7zrl">253.35</td>
    <td class="tg-7zrl">1.16x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">ssd_resnet50_v1_ckpt</td>
    <td class="tg-7zrl">37.81%</td>
    <td class="tg-7zrl">38.00%</td>
    <td class="tg-7zrl">-0.50%</td>
    <td class="tg-7zrl">70.47</td>
    <td class="tg-7zrl">21.79</td>
    <td class="tg-7zrl">3.23x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v1_ckpt</td>
    <td class="tg-7zrl">22.99%</td>
    <td class="tg-7zrl">23.13%</td>
    <td class="tg-7zrl">-0.61%</td>
    <td class="tg-7zrl">852.49</td>
    <td class="tg-7zrl">386.90</td>
    <td class="tg-7zrl">2.20x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">mask_rcnn_inception_v2_ckpt</td>
    <td class="tg-7zrl">28.54%</td>
    <td class="tg-7zrl">28.72%</td>
    <td class="tg-7zrl">-0.63%</td>
    <td class="tg-7zrl">131.43</td>
    <td class="tg-7zrl">51.09</td>
    <td class="tg-7zrl">2.57x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnet50v1.0</td>
    <td class="tg-7zrl">74.11%</td>
    <td class="tg-7zrl">74.27%</td>
    <td class="tg-7zrl">-0.22%</td>
    <td class="tg-7zrl">1543.95</td>
    <td class="tg-7zrl">501.61</td>
    <td class="tg-7zrl">3.08x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">ssd_resnet34</td>
    <td class="tg-7zrl">21.69%</td>
    <td class="tg-7zrl">22.09%</td>
    <td class="tg-7zrl">-1.81%</td>
    <td class="tg-7zrl">43.71</td>
    <td class="tg-7zrl">11.78</td>
    <td class="tg-7zrl">3.71x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">inception_v1</td>
    <td class="tg-7zrl">70.48%</td>
    <td class="tg-7zrl">69.74%</td>
    <td class="tg-7zrl">1.06%</td>
    <td class="tg-7zrl">2227.69</td>
    <td class="tg-7zrl">1051.64</td>
    <td class="tg-7zrl">2.12x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">faster_rcnn_inception_resnet_v2_saved</td>
    <td class="tg-7zrl">37.90%</td>
    <td class="tg-7zrl">38.33%</td>
    <td class="tg-7zrl">-1.12%</td>
    <td class="tg-7zrl">4.05</td>
    <td class="tg-7zrl">2.33</td>
    <td class="tg-7zrl">1.74x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">faster_rcnn_resnet101</td>
    <td class="tg-7zrl">30.28%</td>
    <td class="tg-7zrl">30.39%</td>
    <td class="tg-7zrl">-0.36%</td>
    <td class="tg-7zrl">69.74</td>
    <td class="tg-7zrl">19.90</td>
    <td class="tg-7zrl">3.50x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnetv2_101</td>
    <td class="tg-7zrl">72.67%</td>
    <td class="tg-7zrl">71.87%</td>
    <td class="tg-7zrl">1.11%</td>
    <td class="tg-7zrl">444.06</td>
    <td class="tg-7zrl">329.70</td>
    <td class="tg-7zrl">1.35x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">inception_resnet_v2</td>
    <td class="tg-7zrl">80.44%</td>
    <td class="tg-7zrl">80.40%</td>
    <td class="tg-7zrl">0.05%</td>
    <td class="tg-7zrl">284.40</td>
    <td class="tg-7zrl">143.73</td>
    <td class="tg-7zrl">1.98x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnetv2_152</td>
    <td class="tg-7zrl">73.03%</td>
    <td class="tg-7zrl">72.37%</td>
    <td class="tg-7zrl">0.91%</td>
    <td class="tg-7zrl">319.08</td>
    <td class="tg-7zrl">223.37</td>
    <td class="tg-7zrl">1.43x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">2.7.0</td>
    <td class="tg-7zrl">resnet50_fashion</td>
    <td class="tg-7zrl">77.80%</td>
    <td class="tg-7zrl">78.12%</td>
    <td class="tg-7zrl">-0.41%</td>
    <td class="tg-7zrl">3953.56</td>
    <td class="tg-7zrl">2170.49</td>
    <td class="tg-7zrl">1.82x</td>
  </tr>
</tbody>
</table>


### Intel-tensorflow 1.x models

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">bert_large_squad</td>
    <td class="tg-7zrl">92.42</td>
    <td class="tg-7zrl">92.98</td>
    <td class="tg-7zrl">-0.61%</td>
    <td class="tg-7zrl">25.99</td>
    <td class="tg-7zrl">12.55</td>
    <td class="tg-7zrl">2.07x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">bert_base_mrpc</td>
    <td class="tg-7zrl">86.52%</td>
    <td class="tg-7zrl">86.52%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">266.15</td>
    <td class="tg-7zrl">145.02</td>
    <td class="tg-7zrl">1.84x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnet_v1_50_slim</td>
    <td class="tg-7zrl">76.38%</td>
    <td class="tg-7zrl">75.18%</td>
    <td class="tg-7zrl">1.60%</td>
    <td class="tg-7zrl">1515.24</td>
    <td class="tg-7zrl">409.44</td>
    <td class="tg-7zrl">3.70x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnet_v1_101_slim</td>
    <td class="tg-7zrl">77.52%</td>
    <td class="tg-7zrl">76.40%</td>
    <td class="tg-7zrl">1.47%</td>
    <td class="tg-7zrl">837.49</td>
    <td class="tg-7zrl">224.57</td>
    <td class="tg-7zrl">3.73x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnet_v1_152_slim</td>
    <td class="tg-7zrl">77.08%</td>
    <td class="tg-7zrl">76.81%</td>
    <td class="tg-7zrl">0.35%</td>
    <td class="tg-7zrl">587.75</td>
    <td class="tg-7zrl">152.39</td>
    <td class="tg-7zrl">3.86x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">inception_v1_slim</td>
    <td class="tg-7zrl">70.49%</td>
    <td class="tg-7zrl">69.77%</td>
    <td class="tg-7zrl">1.03%</td>
    <td class="tg-7zrl">1968.87</td>
    <td class="tg-7zrl">803.53</td>
    <td class="tg-7zrl">2.45x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">inception_v2_slim</td>
    <td class="tg-7zrl">74.35%</td>
    <td class="tg-7zrl">73.98%</td>
    <td class="tg-7zrl">0.50%</td>
    <td class="tg-7zrl">1591.25</td>
    <td class="tg-7zrl">658.54</td>
    <td class="tg-7zrl">2.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">inception_v3_slim</td>
    <td class="tg-7zrl">78.32%</td>
    <td class="tg-7zrl">77.99%</td>
    <td class="tg-7zrl">0.42%</td>
    <td class="tg-7zrl">941.48</td>
    <td class="tg-7zrl">285.17</td>
    <td class="tg-7zrl">3.30x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">inception_v4_slim</td>
    <td class="tg-7zrl">80.30%</td>
    <td class="tg-7zrl">80.19%</td>
    <td class="tg-7zrl">0.14%</td>
    <td class="tg-7zrl">512.74</td>
    <td class="tg-7zrl">143.42</td>
    <td class="tg-7zrl">3.58x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">vgg16_slim</td>
    <td class="tg-7zrl">72.78%</td>
    <td class="tg-7zrl">70.89%</td>
    <td class="tg-7zrl">2.67%</td>
    <td class="tg-7zrl">609.29</td>
    <td class="tg-7zrl">151.15</td>
    <td class="tg-7zrl">4.03x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">vgg19_slim</td>
    <td class="tg-7zrl">72.60%</td>
    <td class="tg-7zrl">71.01%</td>
    <td class="tg-7zrl">2.24%</td>
    <td class="tg-7zrl">510.33</td>
    <td class="tg-7zrl">122.87</td>
    <td class="tg-7zrl">4.15x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnetv2_50_slim</td>
    <td class="tg-7zrl">70.47%</td>
    <td class="tg-7zrl">69.72%</td>
    <td class="tg-7zrl">1.08%</td>
    <td class="tg-7zrl">823.59</td>
    <td class="tg-7zrl">470.80</td>
    <td class="tg-7zrl">1.75x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnetv2_101_slim</td>
    <td class="tg-7zrl">72.62%</td>
    <td class="tg-7zrl">71.91%</td>
    <td class="tg-7zrl">0.99%</td>
    <td class="tg-7zrl">471.451</td>
    <td class="tg-7zrl">247.627</td>
    <td class="tg-7zrl">1.90x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">intel-tensorflow</td>
    <td class="tg-7zrl">1.15.0-up3</td>
    <td class="tg-7zrl">resnetv2_152_slim</td>
    <td class="tg-7zrl">72.95%</td>
    <td class="tg-7zrl">72.40%</td>
    <td class="tg-7zrl">0.76%</td>
    <td class="tg-7zrl">339.192</td>
    <td class="tg-7zrl">170.545</td>
    <td class="tg-7zrl">1.99x</td>
  </tr>
</tbody>
</table>


### PyTorch models

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">se_resnext50_32x4d</td>
    <td class="tg-7zrl">79.04%</td>
    <td class="tg-7zrl">79.08%</td>
    <td class="tg-7zrl">-0.05%</td>
    <td class="tg-7zrl">350.90</td>
    <td class="tg-7zrl">171.32</td>
    <td class="tg-7zrl">2.05x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">mobilenet_v2</td>
    <td class="tg-7zrl">70.54%</td>
    <td class="tg-7zrl">71.84%</td>
    <td class="tg-7zrl">-1.81%</td>
    <td class="tg-7zrl">707.15</td>
    <td class="tg-7zrl">490.61</td>
    <td class="tg-7zrl">1.44x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">rnnt</td>
    <td class="tg-7zrl">92.48</td>
    <td class="tg-7zrl">92.54</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">75.74</td>
    <td class="tg-7zrl">20.44</td>
    <td class="tg-7zrl">3.71x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">barthez_mrpc</td>
    <td class="tg-7zrl">82.99%</td>
    <td class="tg-7zrl">83.81%</td>
    <td class="tg-7zrl">-0.97%</td>
    <td class="tg-7zrl">155.80</td>
    <td class="tg-7zrl">89.41</td>
    <td class="tg-7zrl">1.74x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">longformer_mrpc</td>
    <td class="tg-7zrl">90.59%</td>
    <td class="tg-7zrl">91.46%</td>
    <td class="tg-7zrl">-0.95%</td>
    <td class="tg-7zrl">21.29</td>
    <td class="tg-7zrl">17.15</td>
    <td class="tg-7zrl">1.24x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet18</td>
    <td class="tg-7zrl">69.57%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.27%</td>
    <td class="tg-7zrl">749.77</td>
    <td class="tg-7zrl">377.16</td>
    <td class="tg-7zrl">1.99x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet50</td>
    <td class="tg-7zrl">75.98%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">-0.21%</td>
    <td class="tg-7zrl">487.25</td>
    <td class="tg-7zrl">199.64</td>
    <td class="tg-7zrl">2.44x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnext101_32x8d</td>
    <td class="tg-7zrl">79.03%</td>
    <td class="tg-7zrl">79.31%</td>
    <td class="tg-7zrl">-0.35%</td>
    <td class="tg-7zrl">198.94</td>
    <td class="tg-7zrl">73.88</td>
    <td class="tg-7zrl">2.69x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet18_qat</td>
    <td class="tg-7zrl">69.74%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.03%</td>
    <td class="tg-7zrl">750.71</td>
    <td class="tg-7zrl">379.57</td>
    <td class="tg-7zrl">1.98x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet50_qat</td>
    <td class="tg-7zrl">76.04%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">-0.14%</td>
    <td class="tg-7zrl">478.44</td>
    <td class="tg-7zrl">197.69</td>
    <td class="tg-7zrl">2.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">inception_v3</td>
    <td class="tg-7zrl">69.43%</td>
    <td class="tg-7zrl">69.52%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">433.36</td>
    <td class="tg-7zrl">216.31</td>
    <td class="tg-7zrl">2.00x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">peleenet</td>
    <td class="tg-7zrl">71.64%</td>
    <td class="tg-7zrl">72.10%</td>
    <td class="tg-7zrl">-0.64%</td>
    <td class="tg-7zrl">479.00</td>
    <td class="tg-7zrl">377.54</td>
    <td class="tg-7zrl">1.27x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">yolo_v3</td>
    <td class="tg-7zrl">24.60%</td>
    <td class="tg-7zrl">24.54%</td>
    <td class="tg-7zrl">0.21%</td>
    <td class="tg-7zrl">105.84</td>
    <td class="tg-7zrl">39.80</td>
    <td class="tg-7zrl">2.66x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">blendcnn</td>
    <td class="tg-7zrl">68.40%</td>
    <td class="tg-7zrl">68.40%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">4997.74</td>
    <td class="tg-7zrl">4621.03</td>
    <td class="tg-7zrl">1.08x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">roberta_base_mrpc</td>
    <td class="tg-7zrl">87.88%</td>
    <td class="tg-7zrl">88.18%</td>
    <td class="tg-7zrl">-0.34%</td>
    <td class="tg-7zrl">246.27</td>
    <td class="tg-7zrl">125.03</td>
    <td class="tg-7zrl">1.97x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">camembert_base_mrpc</td>
    <td class="tg-7zrl">86.56%</td>
    <td class="tg-7zrl">86.82%</td>
    <td class="tg-7zrl">-0.30%</td>
    <td class="tg-7zrl">236.17</td>
    <td class="tg-7zrl">124.68</td>
    <td class="tg-7zrl">1.89x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">distilbert_base_mrpc</td>
    <td class="tg-7zrl">88.66%</td>
    <td class="tg-7zrl">89.16%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">422.29</td>
    <td class="tg-7zrl">246.37</td>
    <td class="tg-7zrl">1.71x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">albert_base_mrpc</td>
    <td class="tg-7zrl">88.06%</td>
    <td class="tg-7zrl">88.50%</td>
    <td class="tg-7zrl">-0.50%</td>
    <td class="tg-7zrl">34.44</td>
    <td class="tg-7zrl">28.85</td>
    <td class="tg-7zrl">1.19x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">pegasus_samsum</td>
    <td class="tg-7zrl">42.20</td>
    <td class="tg-7zrl">42.67</td>
    <td class="tg-7zrl">-1.09%</td>
    <td class="tg-7zrl">3.80</td>
    <td class="tg-7zrl">1.14</td>
    <td class="tg-7zrl">3.33x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">flaubert_mrpc</td>
    <td class="tg-7zrl">81.01%</td>
    <td class="tg-7zrl">80.19%</td>
    <td class="tg-7zrl">1.01%</td>
    <td class="tg-7zrl">672.25</td>
    <td class="tg-7zrl">457.05</td>
    <td class="tg-7zrl">1.47x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">deberta_mrpc</td>
    <td class="tg-7zrl">91.17%</td>
    <td class="tg-7zrl">90.91%</td>
    <td class="tg-7zrl">0.28%</td>
    <td class="tg-7zrl">131.09</td>
    <td class="tg-7zrl">79.85</td>
    <td class="tg-7zrl">1.64x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">squeezebert_mrpc</td>
    <td class="tg-7zrl">87.77%</td>
    <td class="tg-7zrl">87.65%</td>
    <td class="tg-7zrl">0.14%</td>
    <td class="tg-7zrl">239.56</td>
    <td class="tg-7zrl">209.01</td>
    <td class="tg-7zrl">1.15x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet18_fx</td>
    <td class="tg-7zrl">69.57%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.28%</td>
    <td class="tg-7zrl">761.15</td>
    <td class="tg-7zrl">379.99</td>
    <td class="tg-7zrl">2.00x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet18_qat_fx</td>
    <td class="tg-7zrl">69.73%</td>
    <td class="tg-7zrl">69.76%</td>
    <td class="tg-7zrl">-0.04%</td>
    <td class="tg-7zrl">765.09</td>
    <td class="tg-7zrl">377.01</td>
    <td class="tg-7zrl">2.03x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">transfo_xl_mrpc</td>
    <td class="tg-7zrl">81.97%</td>
    <td class="tg-7zrl">81.20%</td>
    <td class="tg-7zrl">0.94%</td>
    <td class="tg-7zrl">11.10</td>
    <td class="tg-7zrl">8.22</td>
    <td class="tg-7zrl">1.35x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_base_mrpc</td>
    <td class="tg-7zrl">90.28%</td>
    <td class="tg-7zrl">90.69%</td>
    <td class="tg-7zrl">-0.45%</td>
    <td class="tg-7zrl">241.46</td>
    <td class="tg-7zrl">125.09</td>
    <td class="tg-7zrl">1.93x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_base_cola</td>
    <td class="tg-7zrl">58.80%</td>
    <td class="tg-7zrl">58.84%</td>
    <td class="tg-7zrl">-0.07%</td>
    <td class="tg-7zrl">253.12</td>
    <td class="tg-7zrl">125.17</td>
    <td class="tg-7zrl">2.02x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_base_sts-b</td>
    <td class="tg-7zrl">89.13%</td>
    <td class="tg-7zrl">89.75%</td>
    <td class="tg-7zrl">-0.68%</td>
    <td class="tg-7zrl">243.50</td>
    <td class="tg-7zrl">124.54</td>
    <td class="tg-7zrl">1.96x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_base_sst-2</td>
    <td class="tg-7zrl">91.97%</td>
    <td class="tg-7zrl">91.86%</td>
    <td class="tg-7zrl">0.12%</td>
    <td class="tg-7zrl">252.00</td>
    <td class="tg-7zrl">121.14</td>
    <td class="tg-7zrl">2.08x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_large_cola</td>
    <td class="tg-7zrl">62.88%</td>
    <td class="tg-7zrl">62.57%</td>
    <td class="tg-7zrl">0.49%</td>
    <td class="tg-7zrl">87.88</td>
    <td class="tg-7zrl">36.93</td>
    <td class="tg-7zrl">2.38x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_base_rte</td>
    <td class="tg-7zrl">69.31%</td>
    <td class="tg-7zrl">69.68%</td>
    <td class="tg-7zrl">-0.52%</td>
    <td class="tg-7zrl">244.20</td>
    <td class="tg-7zrl">125.71</td>
    <td class="tg-7zrl">1.94x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_large_mrpc</td>
    <td class="tg-7zrl">89.93%</td>
    <td class="tg-7zrl">90.38%</td>
    <td class="tg-7zrl">-0.49%</td>
    <td class="tg-7zrl">87.44</td>
    <td class="tg-7zrl">36.71</td>
    <td class="tg-7zrl">2.38x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_large_qnli</td>
    <td class="tg-7zrl">90.96%</td>
    <td class="tg-7zrl">91.82%</td>
    <td class="tg-7zrl">-0.94%</td>
    <td class="tg-7zrl">89.18</td>
    <td class="tg-7zrl">36.87</td>
    <td class="tg-7zrl">2.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_large_rte</td>
    <td class="tg-7zrl">71.84%</td>
    <td class="tg-7zrl">72.56%</td>
    <td class="tg-7zrl">-1.00%</td>
    <td class="tg-7zrl">75.91</td>
    <td class="tg-7zrl">36.72</td>
    <td class="tg-7zrl">2.07x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">mbart_wnli</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">65.24</td>
    <td class="tg-7zrl">31.06</td>
    <td class="tg-7zrl">2.10x</td>
  </tr>
</tbody>
</table>

### PyTorch models along with ipex

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
<tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnet50_ipex</td>
    <td class="tg-7zrl">76.14%</td>
    <td class="tg-7zrl">76.15%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">654.50</td>
    <td class="tg-7zrl">202.31</td>
    <td class="tg-7zrl">3.24x</td>
  </tr>
<tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">bert_large_ipex</td>
    <td class="tg-7zrl">92.77</td>
    <td class="tg-7zrl">93.16</td>
    <td class="tg-7zrl">-0.41%</td>
    <td class="tg-7zrl">29.74</td>
    <td class="tg-7zrl">13.61</td>
    <td class="tg-7zrl">2.18x</td>
</tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">resnext101_32x16d_wsl_ipex</td>
    <td class="tg-7zrl">84.02%</td>
    <td class="tg-7zrl">84.17%</td>
    <td class="tg-7zrl">-0.18%</td>
    <td class="tg-7zrl">157.78</td>
    <td class="tg-7zrl">28.54</td>
    <td class="tg-7zrl">5.53x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">pytorch</td>
    <td class="tg-7zrl">1.10.0+cpu</td>
    <td class="tg-7zrl">ssd_resnet34_ipex</td>
    <td class="tg-7zrl">19.95%</td>
    <td class="tg-7zrl">20.00%</td>
    <td class="tg-7zrl">-0.25%</td>
    <td class="tg-7zrl">30.50</td>
    <td class="tg-7zrl">8.50</td>
    <td class="tg-7zrl">3.59x</td>
  </tr>
</tbody>
</table>


### MXNet models

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">mxnet</td>
    <td class="tg-7zrl">1.7.0</td>
    <td class="tg-7zrl">inceptionv3</td>
    <td class="tg-7zrl">77.80%</td>
    <td class="tg-7zrl">77.65%</td>
    <td class="tg-7zrl">0.20%</td>
    <td class="tg-7zrl">918.73</td>
    <td class="tg-7zrl">238.90</td>
    <td class="tg-7zrl">3.85x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">mxnet</td>
    <td class="tg-7zrl">1.7.0</td>
    <td class="tg-7zrl">squeezenet1.0</td>
    <td class="tg-7zrl">56.80%</td>
    <td class="tg-7zrl">56.97%</td>
    <td class="tg-7zrl">-0.28%</td>
    <td class="tg-7zrl">4693.55</td>
    <td class="tg-7zrl">1272.50</td>
    <td class="tg-7zrl">3.69x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">mxnet</td>
    <td class="tg-7zrl">1.7.0</td>
    <td class="tg-7zrl">ssd-mobilenet1.0</td>
    <td class="tg-7zrl">74.94%</td>
    <td class="tg-7zrl">75.54%</td>
    <td class="tg-7zrl">-0.79%</td>
    <td class="tg-7zrl">771.65</td>
    <td class="tg-7zrl">189.81</td>
    <td class="tg-7zrl">4.07x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">mxnet</td>
    <td class="tg-7zrl">1.7.0</td>
    <td class="tg-7zrl">resnet152_v1</td>
    <td class="tg-7zrl">78.28%</td>
    <td class="tg-7zrl">78.54%</td>
    <td class="tg-7zrl">-0.33%</td>
    <td class="tg-7zrl">574.23</td>
    <td class="tg-7zrl">126.78</td>
    <td class="tg-7zrl">4.53x</td>
  </tr>
</tbody>
</table>


### ONNX Models

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">alexnet</td>
    <td class="tg-7zrl">54.74%</td>
    <td class="tg-7zrl">54.79%</td>
    <td class="tg-7zrl">-0.09%</td>
    <td class="tg-7zrl">1505.75</td>
    <td class="tg-7zrl">656.81</td>
    <td class="tg-7zrl">2.29x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">zfnet</td>
    <td class="tg-7zrl">55.89%</td>
    <td class="tg-7zrl">55.96%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">661.16</td>
    <td class="tg-7zrl">353.20</td>
    <td class="tg-7zrl">1.87x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">efficientnet</td>
    <td class="tg-7zrl">77.58%</td>
    <td class="tg-7zrl">77.70%</td>
    <td class="tg-7zrl">-0.15%</td>
    <td class="tg-7zrl">2065.72</td>
    <td class="tg-7zrl">1094.77</td>
    <td class="tg-7zrl">1.89x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">squeezenet_qdq</td>
    <td class="tg-7zrl">56.55%</td>
    <td class="tg-7zrl">56.87%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">5965.78</td>
    <td class="tg-7zrl">4300.12</td>
    <td class="tg-7zrl">1.39x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd-12_qdq</td>
    <td class="tg-7zrl">18.38%</td>
    <td class="tg-7zrl">18.98%</td>
    <td class="tg-7zrl">-3.16%</td>
    <td class="tg-7zrl">42.24</td>
    <td class="tg-7zrl">11.12</td>
    <td class="tg-7zrl">3.80x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">resnet50_v1_5</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.29%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1166.31</td>
    <td class="tg-7zrl">554.34</td>
    <td class="tg-7zrl">2.10x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">bert_base_mrpc_static</td>
    <td class="tg-7zrl">85.29%</td>
    <td class="tg-7zrl">86.03%</td>
    <td class="tg-7zrl">-0.86%</td>
    <td class="tg-7zrl">766.46</td>
    <td class="tg-7zrl">315.22</td>
    <td class="tg-7zrl">2.43x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">bert_base_mrpc_dynamic</td>
    <td class="tg-7zrl">85.54%</td>
    <td class="tg-7zrl">86.03%</td>
    <td class="tg-7zrl">-0.57%</td>
    <td class="tg-7zrl">381.30</td>
    <td class="tg-7zrl">155.90</td>
    <td class="tg-7zrl">2.45x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilenet_v2</td>
    <td class="tg-7zrl">65.47%</td>
    <td class="tg-7zrl">66.89%</td>
    <td class="tg-7zrl">-2.12%</td>
    <td class="tg-7zrl">5128.93</td>
    <td class="tg-7zrl">3390.19</td>
    <td class="tg-7zrl">1.51x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v1</td>
    <td class="tg-7zrl">22.20%</td>
    <td class="tg-7zrl">23.10%</td>
    <td class="tg-7zrl">-3.90%</td>
    <td class="tg-7zrl">914.92</td>
    <td class="tg-7zrl">703.74</td>
    <td class="tg-7zrl">1.30x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v2</td>
    <td class="tg-7zrl">23.83%</td>
    <td class="tg-7zrl">24.68%</td>
    <td class="tg-7zrl">-3.44%</td>
    <td class="tg-7zrl">718.28</td>
    <td class="tg-7zrl">501.31</td>
    <td class="tg-7zrl">1.43x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">distilbert_base_mrpc</td>
    <td class="tg-7zrl">84.56%</td>
    <td class="tg-7zrl">84.56%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">1675.94</td>
    <td class="tg-7zrl">594.27</td>
    <td class="tg-7zrl">2.82x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilebert_mrpc</td>
    <td class="tg-7zrl">85.54%</td>
    <td class="tg-7zrl">86.27%</td>
    <td class="tg-7zrl">-0.85%</td>
    <td class="tg-7zrl">766.00</td>
    <td class="tg-7zrl">684.30</td>
    <td class="tg-7zrl">1.12x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">resnet50-v1-12</td>
    <td class="tg-7zrl">74.76%</td>
    <td class="tg-7zrl">74.99%</td>
    <td class="tg-7zrl">-0.31%</td>
    <td class="tg-7zrl">1380.38</td>
    <td class="tg-7zrl">581.36</td>
    <td class="tg-7zrl">2.37x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">resnet_v1_5_mlperf</td>
    <td class="tg-7zrl">76.13%</td>
    <td class="tg-7zrl">76.46%</td>
    <td class="tg-7zrl">-0.43%</td>
    <td class="tg-7zrl">1143.13</td>
    <td class="tg-7zrl">550.77</td>
    <td class="tg-7zrl">2.08x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilenet_v3_mlperf</td>
    <td class="tg-7zrl">75.59%</td>
    <td class="tg-7zrl">75.74%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">4121.33</td>
    <td class="tg-7zrl">2135.31</td>
    <td class="tg-7zrl">1.93x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">shufflenet-v2-12</td>
    <td class="tg-7zrl">66.13%</td>
    <td class="tg-7zrl">66.36%</td>
    <td class="tg-7zrl">-0.35%</td>
    <td class="tg-7zrl">4901.74</td>
    <td class="tg-7zrl">2853.37</td>
    <td class="tg-7zrl">1.72x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">googlenet-12</td>
    <td class="tg-7zrl">67.61%</td>
    <td class="tg-7zrl">67.79%</td>
    <td class="tg-7zrl">-0.27%</td>
    <td class="tg-7zrl">1030.75</td>
    <td class="tg-7zrl">805.76</td>
    <td class="tg-7zrl">1.28x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">squeezenet</td>
    <td class="tg-7zrl">56.55%</td>
    <td class="tg-7zrl">56.87%</td>
    <td class="tg-7zrl">-0.56%</td>
    <td class="tg-7zrl">6119.01</td>
    <td class="tg-7zrl">4321.71</td>
    <td class="tg-7zrl">1.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">caffenet</td>
    <td class="tg-7zrl">56.19%</td>
    <td class="tg-7zrl">56.30%</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">2644.16</td>
    <td class="tg-7zrl">810.13</td>
    <td class="tg-7zrl">3.26x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">inception_v1</td>
    <td class="tg-7zrl">67.23%</td>
    <td class="tg-7zrl">67.24%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1059.31</td>
    <td class="tg-7zrl">848.19</td>
    <td class="tg-7zrl">1.25x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">fcn</td>
    <td class="tg-7zrl">64.66%</td>
    <td class="tg-7zrl">64.98%</td>
    <td class="tg-7zrl">-0.49%</td>
    <td class="tg-7zrl">44.48</td>
    <td class="tg-7zrl">14.23</td>
    <td class="tg-7zrl">3.13x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd-12</td>
    <td class="tg-7zrl">18.84%</td>
    <td class="tg-7zrl">18.98%</td>
    <td class="tg-7zrl">-0.74%</td>
    <td class="tg-7zrl">41.98</td>
    <td class="tg-7zrl">11.11</td>
    <td class="tg-7zrl">3.78x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v1-2</td>
    <td class="tg-7zrl">22.88%</td>
    <td class="tg-7zrl">23.03%</td>
    <td class="tg-7zrl">-0.65%</td>
    <td class="tg-7zrl">836.01</td>
    <td class="tg-7zrl">652.27</td>
    <td class="tg-7zrl">1.28x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">faster_rcnn</td>
    <td class="tg-7zrl">33.99%</td>
    <td class="tg-7zrl">34.37%</td>
    <td class="tg-7zrl">-1.11%</td>
    <td class="tg-7zrl">9.23</td>
    <td class="tg-7zrl">4.28</td>
    <td class="tg-7zrl">2.16x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilenetv2-12</td>
    <td class="tg-7zrl">68.30%</td>
    <td class="tg-7zrl">69.48%</td>
    <td class="tg-7zrl">-1.70%</td>
    <td class="tg-7zrl">5314.59</td>
    <td class="tg-7zrl">3369.52</td>
    <td class="tg-7zrl">1.58x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mask_rcnn</td>
    <td class="tg-7zrl">33.40%</td>
    <td class="tg-7zrl">33.72%</td>
    <td class="tg-7zrl">-0.95%</td>
    <td class="tg-7zrl">7.88</td>
    <td class="tg-7zrl">3.94</td>
    <td class="tg-7zrl">2.00x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">yolov3</td>
    <td class="tg-7zrl">26.88%</td>
    <td class="tg-7zrl">28.74%</td>
    <td class="tg-7zrl">-6.47%</td>
    <td class="tg-7zrl">157.85</td>
    <td class="tg-7zrl">64.93</td>
    <td class="tg-7zrl">2.43x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">densenet</td>
    <td class="tg-7zrl">60.20%</td>
    <td class="tg-7zrl">60.96%</td>
    <td class="tg-7zrl">-1.25%</td>
    <td class="tg-7zrl">408.55</td>
    <td class="tg-7zrl">340.82</td>
    <td class="tg-7zrl">1.20x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">yolov4</td>
    <td class="tg-7zrl">30.95%</td>
    <td class="tg-7zrl">32.78%</td>
    <td class="tg-7zrl">-5.58%</td>
    <td class="tg-7zrl">53.51</td>
    <td class="tg-7zrl">28.66</td>
    <td class="tg-7zrl">1.87x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">resnet50_v1_5_qdq</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.29%</td>
    <td class="tg-7zrl">-0.01%</td>
    <td class="tg-7zrl">1271.61</td>
    <td class="tg-7zrl">543.58</td>
    <td class="tg-7zrl">2.34x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilenet_v2_qdq</td>
    <td class="tg-7zrl">65.47%</td>
    <td class="tg-7zrl">66.89%</td>
    <td class="tg-7zrl">-2.12%</td>
    <td class="tg-7zrl">5069.54</td>
    <td class="tg-7zrl">3404.88</td>
    <td class="tg-7zrl">1.49x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">ssd_mobilenet_v1_qdq</td>
    <td class="tg-7zrl">22.25%</td>
    <td class="tg-7zrl">23.10%</td>
    <td class="tg-7zrl">-3.68%</td>
    <td class="tg-7zrl">803.63</td>
    <td class="tg-7zrl">644.18</td>
    <td class="tg-7zrl">1.25x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">vgg16</td>
    <td class="tg-7zrl">66.60%</td>
    <td class="tg-7zrl">66.69%</td>
    <td class="tg-7zrl">-0.13%</td>
    <td class="tg-7zrl">310.23</td>
    <td class="tg-7zrl">128.81</td>
    <td class="tg-7zrl">2.41x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">roberta_base_mrpc</td>
    <td class="tg-7zrl">89.22%</td>
    <td class="tg-7zrl">89.95%</td>
    <td class="tg-7zrl">-0.81%</td>
    <td class="tg-7zrl">766.66</td>
    <td class="tg-7zrl">316.24</td>
    <td class="tg-7zrl">2.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">bert_squad_model_zoo</td>
    <td class="tg-7zrl">80.43</td>
    <td class="tg-7zrl">80.67</td>
    <td class="tg-7zrl">-0.29%</td>
    <td class="tg-7zrl">115.78</td>
    <td class="tg-7zrl">64.69</td>
    <td class="tg-7zrl">1.79x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">mobilebert_squad_mlperf</td>
    <td class="tg-7zrl">89.84</td>
    <td class="tg-7zrl">90.02</td>
    <td class="tg-7zrl">-0.20%</td>
    <td class="tg-7zrl">102.82</td>
    <td class="tg-7zrl">95.17</td>
    <td class="tg-7zrl">1.08x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">onnxrt-runtime</td>
    <td class="tg-7zrl">1.10.0</td>
    <td class="tg-7zrl">vgg16_model_zoo</td>
    <td class="tg-7zrl">72.28%</td>
    <td class="tg-7zrl">72.40%</td>
    <td class="tg-7zrl">-0.17%</td>
    <td class="tg-7zrl">447.28</td>
    <td class="tg-7zrl">129.59</td>
    <td class="tg-7zrl">3.45x</td>
  </tr>
</tbody>
</table>

### INC-ENGINE Models
<table>
<thead>
  <tr>
    <th rowspan="2">Backend</th>
    <th rowspan="2">model</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance<br>1s4c10ins1bs/throughput<br>(samples/sec)<br></th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Acc Ratio[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_base_mrpc_static</td>
    <td class="tg-7zrl">82.35%</td>
    <td class="tg-7zrl">83.09%</td>
    <td class="tg-7zrl">-0.89%</td>
    <td class="tg-7zrl">497.28</td>
    <td class="tg-7zrl">151.16</td>
    <td class="tg-7zrl">3.29x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">distilroberta_base_wnli_static</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">56.34%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">1097.22</td>
    <td class="tg-7zrl">315.94</td>
    <td class="tg-7zrl">3.47x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">distilbert_base_uncased_emotion_static</td>
    <td class="tg-7zrl">93.90%</td>
    <td class="tg-7zrl">94.20%</td>
    <td class="tg-7zrl">-0.32%</td>
    <td class="tg-7zrl">1081.35</td>
    <td class="tg-7zrl">306.33</td>
    <td class="tg-7zrl">3.53x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_large_squad_static</td>
    <td class="tg-7zrl">9077.66%</td>
    <td class="tg-7zrl">9087.41%</td>
    <td class="tg-7zrl">-0.11%</td>
    <td class="tg-7zrl">49.08</td>
    <td class="tg-7zrl">13.48</td>
    <td class="tg-7zrl">3.64x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">distilbert_base_uncased_mrpc_static</td>
    <td class="tg-7zrl">83.82%</td>
    <td class="tg-7zrl">84.07%</td>
    <td class="tg-7zrl">-0.30%</td>
    <td class="tg-7zrl">1091.99</td>
    <td class="tg-7zrl">303.92</td>
    <td class="tg-7zrl">3.59x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">minilm_l6_h384_uncased_sst2_static</td>
    <td class="tg-7zrl">89.33%</td>
    <td class="tg-7zrl">90.14%</td>
    <td class="tg-7zrl">-0.90%</td>
    <td class="tg-7zrl">2594.77</td>
    <td class="tg-7zrl">1083.84</td>
    <td class="tg-7zrl">2.39x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">roberta_base_mrpc_static</td>
    <td class="tg-7zrl">88.24%</td>
    <td class="tg-7zrl">88.97%</td>
    <td class="tg-7zrl">-0.82%</td>
    <td class="tg-7zrl">508.14</td>
    <td class="tg-7zrl">153.37</td>
    <td class="tg-7zrl">3.31x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_base_nli_mean_tokens_stsb_static</td>
    <td class="tg-7zrl">89.23%</td>
    <td class="tg-7zrl">89.55%</td>
    <td class="tg-7zrl">-0.36%</td>
    <td class="tg-7zrl">546.97</td>
    <td class="tg-7zrl">151.77</td>
    <td class="tg-7zrl">3.60x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">paraphrase_xlm_r_multilingual_v1_stsb_static</td>
    <td class="tg-7zrl">86.66%</td>
    <td class="tg-7zrl">87.23%</td>
    <td class="tg-7zrl">-0.65%</td>
    <td class="tg-7zrl">552.44</td>
    <td class="tg-7zrl">153.74</td>
    <td class="tg-7zrl">3.59x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">distilbert_base_uncased_sst2_static</td>
    <td class="tg-7zrl">90.14%</td>
    <td class="tg-7zrl">90.25%</td>
    <td class="tg-7zrl">-0.12%</td>
    <td class="tg-7zrl">1086.13</td>
    <td class="tg-7zrl">306.45</td>
    <td class="tg-7zrl">3.54x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">finbert_financial_phrasebank_static</td>
    <td class="tg-7zrl">82.57%</td>
    <td class="tg-7zrl">82.80%</td>
    <td class="tg-7zrl">-0.28%</td>
    <td class="tg-7zrl">999.94</td>
    <td class="tg-7zrl">292.55</td>
    <td class="tg-7zrl">3.42x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_base_sparse_mrpc_static</td>
    <td class="tg-7zrl">70.59%</td>
    <td class="tg-7zrl">70.59%</td>
    <td class="tg-7zrl">0.00%</td>
    <td class="tg-7zrl">551.90</td>
    <td class="tg-7zrl">153.80</td>
    <td class="tg-7zrl">3.59x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_mini_mrpc_static</td>
    <td class="tg-7zrl">78.19%</td>
    <td class="tg-7zrl">78.68%</td>
    <td class="tg-7zrl">-0.62%</td>
    <td class="tg-7zrl">6962.58</td>
    <td class="tg-7zrl">3252.14</td>
    <td class="tg-7zrl">2.14x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">bert_mini_sst2_static</td>
    <td class="tg-7zrl">87.16%</td>
    <td class="tg-7zrl">86.93%</td>
    <td class="tg-7zrl">0.26%</td>
    <td class="tg-7zrl">6850.38</td>
    <td class="tg-7zrl">3218.98</td>
    <td class="tg-7zrl">2.13x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">dlrm</td>
    <td class="tg-7zrl">78.09%</td>
    <td class="tg-7zrl">78.10%</td>
    <td class="tg-7zrl">-0.04%</td>
    <td class="tg-7zrl">53412.71</td>
    <td class="tg-7zrl">46422.06</td>
    <td class="tg-7zrl">1.15x</td>
  </tr>
  <tr>
    <td class="tg-7zrl">INC-ENGINE</td>
    <td class="tg-7zrl">dlrm_bf16</td>
    <td class="tg-7zrl">78.10%</td>
    <td class="tg-7zrl">78.09%</td>
    <td class="tg-7zrl">0.01%</td>
    <td class="tg-7zrl">40602.16</td>
    <td class="tg-7zrl">38968.03</td>
    <td class="tg-7zrl">1.04x</td>
  </tr>
</tbody>
</table>

### BACKUP
<table>
<tr><th>System Configuration</th><th>Intel Xeon Platinum 8380 Scalable processor</th></tr>
<tr>
<td>Test Date</td>
<td>Sat 30 Apr 2022 UTC</td>
</tr><tr>
<td>Manufacturer</td>
<td>Intel Corporation</td>
</tr><tr>
<td>Product Name</td>
<td>M50CYP2SBSTD</td>
</tr><tr>
<td>BIOS Version</td>
<td>SE5C6200.86B.0022.D64.2105220049</td>
</tr><tr>
<td>OS</td>
<td>Ubuntu 20.04.1 LTS</td>
</tr><tr>
<td>Kernel</td>
<td>5.4.0-42-generic</td>
</tr><tr>
<td>Microcode</td>
<td>0xd0002b1</td>
</tr><tr>
<td>CPU Model</td>
<td>Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz</td>
</tr><tr>
<td>Base Frequency</td>
<td>2.3GHZ</td>
</tr><tr>
<td>Thread(s) per Core</td>
<td>2</td>
</tr><tr>
<td>Core(s) per Socket</td>
<td>40</td>
</tr><tr>
<td>Socket(s)</td>
<td>2</td>
</tr><tr>
<td>Turbo</td>
<td>Enabled</td>
</tr><tr>
<td>Power & Perf Policy</td>
<td>Balanced</td>
</tr><tr>
<td>Installed</td>
<td>256GB (16x16GB DDR4 3200MT/s [3200MT/s])</td>
</tr><tr>
<td>NIC Summary</td>
<td>2x Ethernet Controller 10G X550T</td>
</tr><tr>
<td>Drive Summary</td>
<td>1x INTEL_SSDSC2KW01 953.9G,
1x CT1000MX500SSD1  931.5G,
1x CT1000MX500SSD1  931.5G
</td>
</tr><tr>
</table>

## Validated Pruning Models
<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">FWK</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">fp32 baseline</th>
    <th colspan="3">gradient sensitivity with 20% sparsity</th>
    <th colspan="3">+onnx dynamic quantization on pruned model</th>
  </tr>
  <tr>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>perf gain (sample/s)</td>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>perf gain (sample/s)</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SST-2</td>
    <td>pytorch</td>
    <td>bert-base</td>
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
    <td>pytorch</td>
    <td>bert-base</td>
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
    <th rowspan="2">FWK</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">fp32 baseline</th>
    <th colspan="2">Pattern Lock on 70% Unstructured Sparsity</th>
    <th colspan="2">Pattern Lock on 50% 1:2 Structured Sparsity</th>
  </tr>
  <tr>
    <td>accuracy%</td>
    <td> drop%</td>
    <td>accuracy%</td>
    <td> drop%</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MNLI</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[m, mm] = [84.57, 84.79]</td>
    <td>[m, mm] = [82.45, 83.27]</td>
    <td>[-2.51, -1.80]</td>
    <td>[m, mm] = [83.20, 84.11]</td>
    <td>[-1.62, -0.80]</td>
  </tr>
  <tr>
    <td>SST-2</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>accuracy = 92.32</td>
    <td>accuracy = 91.51</td>
    <td>-0.88</td>
    <td>accuracy = 92.20</td>
    <td>-0.13</td>
  </tr>
  <tr>
    <td>QQP</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>[accuracy, f1] = [91.10, 88.05]</td>
    <td>[accuracy, f1] = [90.48, 87.06]</td>
    <td>[-0.68, -1.12]</td>
    <td>[accuracy, f1] = [90.92, 87.78]</td>
    <td>[-0.20, -0.31]</td>
  </tr>
  <tr>
    <td>QNLI</td>
    <td>pytorch</td>
    <td>bert-base</td>
    <td>accuracy = 91.54</td>
    <td>accuracy = 90.39</td>
    <td>-1.26</td>
    <td>accuracy = 90.87</td>
    <td>-0.73</td>
  </tr>
  <tr>
    <td>QnA</td>
    <td>pytorch</td>
    <td>bert-base</td>
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
    <th>fp32 baseline</th>
    <th>Compression</th>
    <th>dataset</th>
    <th>acc(drop)%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Pytorch</td>
    <td>resnet18</td>
    <td>69.76</td>
    <td>30% sparsity on magnitude</td>
    <td>ImageNet</td>
    <td>69.47(-0.42)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet18</td>
    <td>69.76</td>
    <td>30% sparsity on gradient sensitivity</td>
    <td>ImageNet</td>
    <td>68.85(-1.30)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude</td>
    <td>ImageNet</td>
    <td>76.11(-0.03)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude and post training quantization</td>
    <td>ImageNet</td>
    <td>76.01(-0.16)</td>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>resnet50</td>
    <td>76.13</td>
    <td>30% sparsity on magnitude and quantization aware training</td>
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
    <td rowspan="2">BlendCnn example</td>
    <td rowspan="2">MRPC</td>
    <td rowspan="2">BlendCnn<br>(0.7034)</td>
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
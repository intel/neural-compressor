
Validated Models
======

1. [Validated Quantization Examples](#Validated-Quantization-Examples)

    1.1. [TensorFlow Models with TensorFlow 2.11.0](#tensorflow-models-with-tensorflow-2100)

    1.2. [PyTorch Models with Torch 1.13.1+cpu in PTQ Mode](#pytorch-models-with-torch-1131cpu-in-qat-mode)

    1.3. [PyTorch Models with Torch 1.13.1+cpu in QAT Mode](#pytorch-models-with-torch-1131cpu-in-qat-mode)

    1.4. [PyTorch Models with Torch and Intel® Extension for PyTorch* 1.13.0+cpu](#pytorch-models-with-torch-and-intel-extension-for-pytorch-1130cpu)
    
    1.5. [ONNX Models with ONNX Runtime 1.13.1](#onnx-models-with-onnx-runtime-1131)

    1.6. [MXNet Models with MXNet 1.9.1](#mxnet-models-with-mxnet-191)

2. [Validated Pruning Examples](#Validated-Pruning-Examples)

3. [Validated Knowledge Distillation Examples](#Validated-Knowledge-Distillation-Examples)

4. [Validated ONNX QDQ INT8 Models on Multiple Hardware through ONNX Runtime](#validated-onnx-qdq-int8-models-on-multiple-hardware-through-onnx-runtime)

## Validated Quantization Examples

Performance results test on ​​09/24/2022 with Intel Xeon Platinum 8380 Scalable processor, using 1 socket, 4 cores/instance, 8 instances and batch size 1. 

Performance varies by use, configuration and other factors. See [platform configuration](./platform_configuration.md) for configuration details. For more complete information about performance and benchmark results, visit www.intel.com/benchmarks

### TensorFlow Models with TensorFlow 2.11.0

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" rowspan="2">example</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">bert base mrpc</td>
    <td class="tg-8d8j">ckpt</td>
    <td class="tg-8d8j">86.52%</td>
    <td class="tg-8d8j">86.52%</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">170.443</td>
    <td class="tg-8d8j">93.685</td>
    <td class="tg-8d8j">1.82x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large squad</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">92.404</td>
    <td class="tg-8d8j">92.9861</td>
    <td class="tg-8d8j">-0.0063</td>
    <td class="tg-8d8j">18.388</td>
    <td class="tg-8d8j">9.924</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large squad model zoo</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">92.4128</td>
    <td class="tg-8d8j">92.9805</td>
    <td class="tg-8d8j">-0.0061</td>
    <td class="tg-8d8j">20.414</td>
    <td class="tg-8d8j">11.156</td>
    <td class="tg-8d8j">1.83x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">densenet 121</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">73.61%</td>
    <td class="tg-8d8j">72.89%</td>
    <td class="tg-8d8j">0.0099</td>
    <td class="tg-8d8j">274.608</td>
    <td class="tg-8d8j">148.722</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">densenet 161</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">76.30%</td>
    <td class="tg-8d8j">76.29%</td>
    <td class="tg-8d8j">0.0001</td>
    <td class="tg-8d8j">132.345</td>
    <td class="tg-8d8j">95.241</td>
    <td class="tg-8d8j">1.39x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">densenet 169</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">74.38%</td>
    <td class="tg-8d8j">74.65%</td>
    <td class="tg-8d8j">-0.0036</td>
    <td class="tg-8d8j">191.311</td>
    <td class="tg-8d8j">118.987</td>
    <td class="tg-8d8j">1.61x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn inception resnet&nbsp;&nbsp;&nbsp;v2</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">37.44%</td>
    <td class="tg-8d8j">38.31%</td>
    <td class="tg-8d8j">-0.0227</td>
    <td class="tg-8d8j">3.312</td>
    <td class="tg-8d8j">1.813</td>
    <td class="tg-8d8j">1.83x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn inception resnet&nbsp;&nbsp;&nbsp;v2 </td>
    <td class="tg-nrix">SavedModel</td>
    <td class="tg-8d8j">37.55%</td>
    <td class="tg-8d8j">38.31%</td>
    <td class="tg-8d8j">-0.0198</td>
    <td class="tg-8d8j">3.322</td>
    <td class="tg-8d8j">1.809</td>
    <td class="tg-8d8j">1.84x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn resnet101</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">30.33%</td>
    <td class="tg-8d8j">30.39%</td>
    <td class="tg-8d8j">-0.002</td>
    <td class="tg-8d8j">42.568</td>
    <td class="tg-8d8j">13.25</td>
    <td class="tg-8d8j">3.21x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn resnet101 saved</td>
    <td class="tg-nrix">SavedModel</td>
    <td class="tg-8d8j">30.33%</td>
    <td class="tg-8d8j">30.39%</td>
    <td class="tg-8d8j">-0.002</td>
    <td class="tg-8d8j">43.413</td>
    <td class="tg-8d8j">11.733</td>
    <td class="tg-8d8j">3.70x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn resnet50</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">26.64%</td>
    <td class="tg-8d8j">26.59%</td>
    <td class="tg-8d8j">0.0019</td>
    <td class="tg-8d8j">51.704</td>
    <td class="tg-8d8j">16.446</td>
    <td class="tg-8d8j">3.14x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception resnet v2</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">80.34%</td>
    <td class="tg-8d8j">80.40%</td>
    <td class="tg-8d8j">-0.0007</td>
    <td class="tg-8d8j">139.294</td>
    <td class="tg-8d8j">76.653</td>
    <td class="tg-8d8j">1.82x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception resnet v2 </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">80.35%</td>
    <td class="tg-8d8j">80.40%</td>
    <td class="tg-8d8j">-0.0005</td>
    <td class="tg-8d8j">99.424</td>
    <td class="tg-8d8j">54.5</td>
    <td class="tg-8d8j">1.82x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v1</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">70.44%</td>
    <td class="tg-8d8j">69.74%</td>
    <td class="tg-8d8j">0.01</td>
    <td class="tg-8d8j">955.202</td>
    <td class="tg-8d8j">328.148</td>
    <td class="tg-8d8j">2.91x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v2</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">74.34%</td>
    <td class="tg-8d8j">73.97%</td>
    <td class="tg-8d8j">0.005</td>
    <td class="tg-8d8j">709.916</td>
    <td class="tg-8d8j">282.403</td>
    <td class="tg-8d8j">2.51x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v3</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">76.71%</td>
    <td class="tg-8d8j">76.75%</td>
    <td class="tg-8d8j">-0.0005</td>
    <td class="tg-8d8j">337.094</td>
    <td class="tg-8d8j">160.065</td>
    <td class="tg-8d8j">2.11x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v3 </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">77.73%</td>
    <td class="tg-8d8j">77.83%</td>
    <td class="tg-8d8j">-0.0013</td>
    <td class="tg-8d8j">438.515</td>
    <td class="tg-8d8j">204.757</td>
    <td class="tg-8d8j">2.14x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v4</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">80.18%</td>
    <td class="tg-8d8j">80.27%</td>
    <td class="tg-8d8j">-0.0011</td>
    <td class="tg-8d8j">223.024</td>
    <td class="tg-8d8j">105.436</td>
    <td class="tg-8d8j">2.12x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mask rcnn inception v2</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">28.50%</td>
    <td class="tg-8d8j">28.73%</td>
    <td class="tg-8d8j">-0.008</td>
    <td class="tg-8d8j">69.419</td>
    <td class="tg-8d8j">32.997</td>
    <td class="tg-8d8j">2.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mask rcnn inception v2 </td>
    <td class="tg-8d8j">ckpt</td>
    <td class="tg-8d8j">28.50%</td>
    <td class="tg-8d8j">28.73%</td>
    <td class="tg-8d8j">-0.008</td>
    <td class="tg-8d8j">69.467</td>
    <td class="tg-8d8j">32.879</td>
    <td class="tg-8d8j">2.11x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v1</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">71.85%</td>
    <td class="tg-8d8j">70.96%</td>
    <td class="tg-8d8j">0.0125</td>
    <td class="tg-8d8j">1347.654</td>
    <td class="tg-8d8j">439.052</td>
    <td class="tg-8d8j">3.07x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">72.56%</td>
    <td class="tg-8d8j">71.76%</td>
    <td class="tg-8d8j">0.0111</td>
    <td class="tg-8d8j">1192.006</td>
    <td class="tg-8d8j">492.922</td>
    <td class="tg-8d8j">2.42x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2  </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">71.10%</td>
    <td class="tg-8d8j">71.76%</td>
    <td class="tg-8d8j">-0.0091</td>
    <td class="tg-8d8j">412.752</td>
    <td class="tg-8d8j">376.336</td>
    <td class="tg-8d8j">1.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v3</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">74.00%</td>
    <td class="tg-8d8j">75.31%</td>
    <td class="tg-8d8j">-0.0174</td>
    <td class="tg-8d8j">662.066</td>
    <td class="tg-8d8j">397.693</td>
    <td class="tg-8d8j">1.66x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet101</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">77.50%</td>
    <td class="tg-8d8j">76.45%</td>
    <td class="tg-8d8j">0.0137</td>
    <td class="tg-8d8j">299.233</td>
    <td class="tg-8d8j">154.672</td>
    <td class="tg-8d8j">1.93x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet101 </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">61.38%</td>
    <td class="tg-8d8j">61.47%</td>
    <td class="tg-8d8j">-0.0016</td>
    <td class="tg-8d8j">476.394</td>
    <td class="tg-8d8j">227.242</td>
    <td class="tg-8d8j">2.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50 fashion</td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">78.04%</td>
    <td class="tg-8d8j">78.12%</td>
    <td class="tg-8d8j">-0.001</td>
    <td class="tg-8d8j">2734.432</td>
    <td class="tg-8d8j">1299.729</td>
    <td class="tg-8d8j">2.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50 v1.0</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">74.12%</td>
    <td class="tg-8d8j">74.27%</td>
    <td class="tg-8d8j">-0.002</td>
    <td class="tg-8d8j">498.756</td>
    <td class="tg-8d8j">178.724</td>
    <td class="tg-8d8j">2.79x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50 v1.5</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">76.23%</td>
    <td class="tg-8d8j">76.46%</td>
    <td class="tg-8d8j">-0.003</td>
    <td class="tg-8d8j">427.455</td>
    <td class="tg-8d8j">173.249</td>
    <td class="tg-8d8j">2.47x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnetv2 101</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">72.65%</td>
    <td class="tg-8d8j">71.87%</td>
    <td class="tg-8d8j">0.0109</td>
    <td class="tg-8d8j">194.112</td>
    <td class="tg-8d8j">146.42</td>
    <td class="tg-8d8j">1.33x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnetv2 101 </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">71.48%</td>
    <td class="tg-8d8j">71.57%</td>
    <td class="tg-8d8j">-0.0012</td>
    <td class="tg-8d8j">237.088</td>
    <td class="tg-8d8j">187.244</td>
    <td class="tg-8d8j">1.27x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnetv2 152</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">73.07%</td>
    <td class="tg-8d8j">72.37%</td>
    <td class="tg-8d8j">0.0097</td>
    <td class="tg-8d8j">155.044</td>
    <td class="tg-8d8j">112.014</td>
    <td class="tg-8d8j">1.38x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnetv2 50</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">70.44%</td>
    <td class="tg-8d8j">69.64%</td>
    <td class="tg-8d8j">0.0115</td>
    <td class="tg-8d8j">302.546</td>
    <td class="tg-8d8j">215.496</td>
    <td class="tg-8d8j">1.40x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnetv 2 50 </td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">69.20%</td>
    <td class="tg-8d8j">69.03%</td>
    <td class="tg-8d8j">0.0025</td>
    <td class="tg-8d8j">346.988</td>
    <td class="tg-8d8j">312.153</td>
    <td class="tg-8d8j">1.11x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet v1</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">23.12%</td>
    <td class="tg-8d8j">23.13%</td>
    <td class="tg-8d8j">-0.0004</td>
    <td class="tg-8d8j">277.099</td>
    <td class="tg-8d8j">173.609</td>
    <td class="tg-8d8j">1.60x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet v1 </td>
    <td class="tg-8d8j">ckpt</td>
    <td class="tg-8d8j">23.10%</td>
    <td class="tg-8d8j">23.13%</td>
    <td class="tg-8d8j">-0.0013</td>
    <td class="tg-8d8j">273.51</td>
    <td class="tg-8d8j">118.456</td>
    <td class="tg-8d8j">2.31x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd resnet34</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">21.70%</td>
    <td class="tg-8d8j">22.09%</td>
    <td class="tg-8d8j">-0.0177</td>
    <td class="tg-8d8j">33.951</td>
    <td class="tg-8d8j">8.81</td>
    <td class="tg-8d8j">3.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd resnet50 v1</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">37.75%</td>
    <td class="tg-8d8j">38.00%</td>
    <td class="tg-8d8j">-0.0066</td>
    <td class="tg-8d8j">34.108</td>
    <td class="tg-8d8j">15.665</td>
    <td class="tg-8d8j">2.18x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd resnet50 v1 </td>
    <td class="tg-8d8j">ckpt</td>
    <td class="tg-8d8j">37.82%</td>
    <td class="tg-8d8j">38.00%</td>
    <td class="tg-8d8j">-0.0047</td>
    <td class="tg-8d8j">34.566</td>
    <td class="tg-8d8j">13.677</td>
    <td class="tg-8d8j">2.53x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">transformer lt mlperf</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">27.11589</td>
    <td class="tg-8d8j">27.16596</td>
    <td class="tg-8d8j">-0.0018</td>
    <td class="tg-8d8j">3.255</td>
    <td class="tg-8d8j">2.632</td>
    <td class="tg-8d8j">1.24x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">vgg16</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">72.64%</td>
    <td class="tg-8d8j">70.89%</td>
    <td class="tg-8d8j">0.0247</td>
    <td class="tg-8d8j">219.106</td>
    <td class="tg-8d8j">91.302</td>
    <td class="tg-8d8j">2.40x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">vgg19</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">72.69%</td>
    <td class="tg-8d8j">71.01%</td>
    <td class="tg-8d8j">0.0237</td>
    <td class="tg-8d8j">193.606</td>
    <td class="tg-8d8j">78.467</td>
    <td class="tg-8d8j">2.47x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">wide deep large ds</td>
    <td class="tg-8d8j">PB</td>
    <td class="tg-8d8j">77.75%</td>
    <td class="tg-8d8j">77.67%</td>
    <td class="tg-8d8j">0.001</td>
    <td class="tg-8d8j">11506.91</td>
    <td class="tg-8d8j">9665.067</td>
    <td class="tg-8d8j">1.19x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">xception</td>
    <td class="tg-8d8j">keras</td>
    <td class="tg-8d8j">78.43%</td>
    <td class="tg-8d8j">78.94%</td>
    <td class="tg-8d8j">-0.0065</td>
    <td class="tg-8d8j">262.828</td>
    <td class="tg-8d8j">137.354</td>
    <td class="tg-8d8j">1.91x</td>
  </tr>
</tbody>
</table>

### PyTorch Models with Torch 1.13.1+cpu in PTQ Mode

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" rowspan="2">example</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">albert base mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">88.85%</td>
    <td class="tg-8d8j">88.50%</td>
    <td class="tg-8d8j">0.40%</td>
    <td class="tg-8d8j">25.676</td>
    <td class="tg-8d8j">21.579</td>
    <td class="tg-8d8j">1.19x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">barthez mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">83.92%</td>
    <td class="tg-8d8j">83.81%</td>
    <td class="tg-8d8j">0.14%</td>
    <td class="tg-8d8j">143.369</td>
    <td class="tg-8d8j">70.959</td>
    <td class="tg-8d8j">2.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base cola</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">58.80%</td>
    <td class="tg-8d8j">58.84%</td>
    <td class="tg-8d8j">-0.0007</td>
    <td class="tg-8d8j">223.51</td>
    <td class="tg-8d8j">101.394</td>
    <td class="tg-8d8j">2.20x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base mrpc</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">89.90%</td>
    <td class="tg-8d8j">90.69%</td>
    <td class="tg-8d8j">-0.0088</td>
    <td class="tg-8d8j">209.801</td>
    <td class="tg-8d8j">100.956</td>
    <td class="tg-8d8j">2.08x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base rte</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">69.31%</td>
    <td class="tg-8d8j">69.68%</td>
    <td class="tg-8d8j">-0.0052</td>
    <td class="tg-8d8j">221.92</td>
    <td class="tg-8d8j">101.364</td>
    <td class="tg-8d8j">2.19x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base sst-2</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">91.06%</td>
    <td class="tg-8d8j">91.86%</td>
    <td class="tg-8d8j">-0.0087</td>
    <td class="tg-8d8j">224.19</td>
    <td class="tg-8d8j">101.233</td>
    <td class="tg-8d8j">2.21x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base sts-b</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">89.10%</td>
    <td class="tg-8d8j">89.75%</td>
    <td class="tg-8d8j">-0.0072</td>
    <td class="tg-8d8j">218.037</td>
    <td class="tg-8d8j">101.154</td>
    <td class="tg-8d8j">2.16x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large cola</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">64.12%</td>
    <td class="tg-8d8j">62.57%</td>
    <td class="tg-8d8j">0.0248</td>
    <td class="tg-8d8j">75.423</td>
    <td class="tg-8d8j">29.318</td>
    <td class="tg-8d8j">2.57x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large mrpc</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">89.50%</td>
    <td class="tg-8d8j">90.38%</td>
    <td class="tg-8d8j">-0.0097</td>
    <td class="tg-8d8j">75.096</td>
    <td class="tg-8d8j">29.411</td>
    <td class="tg-8d8j">2.55x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large qnli</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">90.90%</td>
    <td class="tg-8d8j">91.82%</td>
    <td class="tg-8d8j">-0.01</td>
    <td class="tg-8d8j">74.804</td>
    <td class="tg-8d8j">29.17</td>
    <td class="tg-8d8j">2.56x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large RTE</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">72.20%</td>
    <td class="tg-8d8j">74.01%</td>
    <td class="tg-8d8j">-2.44%</td>
    <td class="tg-8d8j">40.38</td>
    <td class="tg-8d8j">29.282</td>
    <td class="tg-8d8j">1.38x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert large squad</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">92.61422</td>
    <td class="tg-8d8j">93.15842</td>
    <td class="tg-8d8j">-0.0058</td>
    <td class="tg-8d8j">18.529</td>
    <td class="tg-8d8j">9.818</td>
    <td class="tg-8d8j">1.89x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">blendcnn</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">68.40%</td>
    <td class="tg-8d8j">68.40%</td>
    <td class="tg-8d8j">0</td>
    <td class="tg-8d8j">4885.6</td>
    <td class="tg-8d8j">3715.36</td>
    <td class="tg-8d8j">1.31x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">camembert base mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">86.70%</td>
    <td class="tg-8d8j">86.82%</td>
    <td class="tg-8d8j">-0.14%</td>
    <td class="tg-8d8j">206.002</td>
    <td class="tg-8d8j">98.504</td>
    <td class="tg-8d8j">2.09x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ctrl mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">81.87%</td>
    <td class="tg-8d8j">82.00%</td>
    <td class="tg-8d8j">-0.15%</td>
    <td class="tg-8d8j">19.391</td>
    <td class="tg-8d8j">7.189</td>
    <td class="tg-8d8j">2.70x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">deberta mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">90.88%</td>
    <td class="tg-8d8j">90.91%</td>
    <td class="tg-8d8j">-0.04%</td>
    <td class="tg-8d8j">125.415</td>
    <td class="tg-8d8j">67.674</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">distilbert base mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">88.23%</td>
    <td class="tg-8d8j">89.16%</td>
    <td class="tg-8d8j">-1.05%</td>
    <td class="tg-8d8j">366.274</td>
    <td class="tg-8d8j">197.764</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">distilbert base mrpc</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">88.54%</td>
    <td class="tg-8d8j">89.16%</td>
    <td class="tg-8d8j">-0.0069</td>
    <td class="tg-8d8j">399.63</td>
    <td class="tg-8d8j">197.47</td>
    <td class="tg-8d8j">2.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">flaubert mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">79.87%</td>
    <td class="tg-8d8j">80.19%</td>
    <td class="tg-8d8j">-0.40%</td>
    <td class="tg-8d8j">592.529</td>
    <td class="tg-8d8j">385.005</td>
    <td class="tg-8d8j">1.54x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">gpt j wikitext</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">3.3587</td>
    <td class="tg-8d8j">2.33509</td>
    <td class="tg-8d8j">0.4384</td>
    <td class="tg-8d8j">0.519</td>
    <td class="tg-8d8j">0.2</td>
    <td class="tg-8d8j">2.60x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hubert</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">97.63%</td>
    <td class="tg-8d8j">97.84%</td>
    <td class="tg-8d8j">-0.0021</td>
    <td class="tg-8d8j">9.999</td>
    <td class="tg-8d8j">7.256</td>
    <td class="tg-8d8j">1.38x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v3</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">69.43%</td>
    <td class="tg-8d8j">69.52%</td>
    <td class="tg-8d8j">-0.0013</td>
    <td class="tg-8d8j">446.654</td>
    <td class="tg-8d8j">181.408</td>
    <td class="tg-8d8j">2.46x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">layoutlm mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">81.22%</td>
    <td class="tg-8d8j">78.01%</td>
    <td class="tg-8d8j">4.12%</td>
    <td class="tg-8d8j">204.218</td>
    <td class="tg-8d8j">96.26</td>
    <td class="tg-8d8j">2.12x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">longformer mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">91.01%</td>
    <td class="tg-8d8j">91.46%</td>
    <td class="tg-8d8j">-0.49%</td>
    <td class="tg-8d8j">18.684</td>
    <td class="tg-8d8j">14.246</td>
    <td class="tg-8d8j">1.31x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">maskrcnn</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">37.60%</td>
    <td class="tg-8d8j">37.80%</td>
    <td class="tg-8d8j">-0.53%</td>
    <td class="tg-8d8j">7.195</td>
    <td class="tg-8d8j">4.7708</td>
    <td class="tg-8d8j">1.51x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mbart wnli</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">56.34%</td>
    <td class="tg-8d8j">56.34%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">56.318</td>
    <td class="tg-8d8j">24.766</td>
    <td class="tg-8d8j">2.27x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">70.54%</td>
    <td class="tg-8d8j">71.84%</td>
    <td class="tg-8d8j">-1.81%</td>
    <td class="tg-8d8j">625.379</td>
    <td class="tg-8d8j">451.249</td>
    <td class="tg-8d8j">1.39x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">pegasus samsum</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">42.096</td>
    <td class="tg-8d8j">42.6716</td>
    <td class="tg-8d8j">-0.0135</td>
    <td class="tg-8d8j">3.583</td>
    <td class="tg-8d8j">1.059</td>
    <td class="tg-8d8j">3.38x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">peleenet</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">71.64%</td>
    <td class="tg-8d8j">72.10%</td>
    <td class="tg-8d8j">-0.0064</td>
    <td class="tg-8d8j">402.328</td>
    <td class="tg-8d8j">312.37</td>
    <td class="tg-8d8j">1.29x</td>
  </tr>
  <tr>
    <td class="tg-nrix">pokemon diffusers</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-nrix">275.8026</td>
    <td class="tg-nrix">334.4761</td>
    <td class="tg-nrix">-17.54%</td>
    <td class="tg-nrix">0.0322</td>
    <td class="tg-nrix">0.0217</td>
    <td class="tg-nrix">1.48x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">reformer crime and punishment</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">1.87979</td>
    <td class="tg-8d8j">1.8717</td>
    <td class="tg-8d8j">0.0043</td>
    <td class="tg-8d8j">162.342</td>
    <td class="tg-8d8j">153.645</td>
    <td class="tg-8d8j">1.06x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet18</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">69.57%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.0027</td>
    <td class="tg-8d8j">657.721</td>
    <td class="tg-8d8j">327.694</td>
    <td class="tg-8d8j">2.01x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet18</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">69.62%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.20%</td>
    <td class="tg-8d8j">812.991</td>
    <td class="tg-8d8j">344.985</td>
    <td class="tg-8d8j">2.36x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">75.98%</td>
    <td class="tg-8d8j">76.15%</td>
    <td class="tg-8d8j">-0.0021</td>
    <td class="tg-8d8j">360.161</td>
    <td class="tg-8d8j">161.441</td>
    <td class="tg-8d8j">2.23x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnext101 32x8d</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">79.08%</td>
    <td class="tg-8d8j">79.31%</td>
    <td class="tg-8d8j">-0.0029</td>
    <td class="tg-8d8j">182.838</td>
    <td class="tg-8d8j">60.553</td>
    <td class="tg-8d8j">3.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">roberta base mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">88.25%</td>
    <td class="tg-8d8j">88.18%</td>
    <td class="tg-8d8j">0.08%</td>
    <td class="tg-8d8j">207.407</td>
    <td class="tg-8d8j">98.707</td>
    <td class="tg-8d8j">2.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">squeezebert mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">86.87%</td>
    <td class="tg-8d8j">87.65%</td>
    <td class="tg-8d8j">-0.89%</td>
    <td class="tg-8d8j">195.001</td>
    <td class="tg-8d8j">150.091</td>
    <td class="tg-8d8j">1.30x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd resnet34</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">19.468</td>
    <td class="tg-8d8j">19.63</td>
    <td class="tg-8d8j">-0.83%</td>
    <td class="tg-8d8j">18.564</td>
    <td class="tg-8d8j">6.753</td>
    <td class="tg-8d8j">2.75x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">transfo xl mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">81.97%</td>
    <td class="tg-8d8j">81.20%</td>
    <td class="tg-8d8j">0.94%</td>
    <td class="tg-8d8j">9.728</td>
    <td class="tg-8d8j">6.917</td>
    <td class="tg-8d8j">1.41x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">wav2vec2</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">95.71%</td>
    <td class="tg-8d8j">96.60%</td>
    <td class="tg-8d8j">-0.0092</td>
    <td class="tg-8d8j">23.78</td>
    <td class="tg-8d8j">19.453</td>
    <td class="tg-8d8j">1.22x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">xlm roberta mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">88.24%</td>
    <td class="tg-8d8j">88.24%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">102.191</td>
    <td class="tg-8d8j">102.576</td>
    <td class="tg-8d8j">1.00x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">xlm-roberta-base mrpc</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">88.03%</td>
    <td class="tg-8d8j">88.62%</td>
    <td class="tg-8d8j">-0.67%</td>
    <td class="tg-8d8j">115.163</td>
    <td class="tg-8d8j">98.747</td>
    <td class="tg-8d8j">1.17x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">yolo v3</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">24.60%</td>
    <td class="tg-8d8j">24.54%</td>
    <td class="tg-8d8j">0.21%</td>
    <td class="tg-8d8j">76.145</td>
    <td class="tg-8d8j">31.802</td>
    <td class="tg-8d8j">2.39x</td>
  </tr>
</tbody>
</table>

### PyTorch Models with Torch 1.13.1+cpu in QAT Mode

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" rowspan="2">example</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">bert base MRPC</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">89.20%</td>
    <td class="tg-8d8j">89.50%</td>
    <td class="tg-8d8j">-0.34%</td>
    <td class="tg-8d8j">232.164</td>
    <td class="tg-8d8j">101.885</td>
    <td class="tg-8d8j">2.28x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 18</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">69.68%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.0012</td>
    <td class="tg-8d8j">664.993</td>
    <td class="tg-8d8j">329.146</td>
    <td class="tg-8d8j">2.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 18</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">69.84%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">0.12%</td>
    <td class="tg-8d8j">832.321</td>
    <td class="tg-8d8j">338.475</td>
    <td class="tg-8d8j">2.46x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 50</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">76.03%</td>
    <td class="tg-8d8j">76.15%</td>
    <td class="tg-8d8j">-0.0015</td>
    <td class="tg-8d8j">433.831</td>
    <td class="tg-8d8j">164.977</td>
    <td class="tg-8d8j">2.63x</td>
  </tr>
</tbody>
</table>

### PyTorch Models with Torch and Intel® Extension for PyTorch* 1.13.0+cpu

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" rowspan="2">example</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">bert&nbsp;&nbsp;&nbsp;base MRPC</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">89.20%</td>
    <td class="tg-8d8j">89.50%</td>
    <td class="tg-8d8j">-0.34%</td>
    <td class="tg-8d8j">232.164</td>
    <td class="tg-8d8j">101.885</td>
    <td class="tg-8d8j">2.28x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 18</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">69.68%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">-0.0012</td>
    <td class="tg-8d8j">664.993</td>
    <td class="tg-8d8j">329.146</td>
    <td class="tg-8d8j">2.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 18</td>
    <td class="tg-8d8j">fx</td>
    <td class="tg-8d8j">69.84%</td>
    <td class="tg-8d8j">69.76%</td>
    <td class="tg-8d8j">0.12%</td>
    <td class="tg-8d8j">832.321</td>
    <td class="tg-8d8j">338.475</td>
    <td class="tg-8d8j">2.46x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 50</td>
    <td class="tg-8d8j">eager</td>
    <td class="tg-8d8j">76.03%</td>
    <td class="tg-8d8j">76.15%</td>
    <td class="tg-8d8j">-0.0015</td>
    <td class="tg-8d8j">433.831</td>
    <td class="tg-8d8j">164.977</td>
    <td class="tg-8d8j">2.63x</td>
  </tr>
</tbody>
</table>

### ONNX Models with ONNX Runtime 1.13.1

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" rowspan="2">example</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">alexnet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">54.73%</td>
    <td class="tg-8d8j">54.79%</td>
    <td class="tg-8d8j">-0.11%</td>
    <td class="tg-8d8j">968.215</td>
    <td class="tg-8d8j">473.307</td>
    <td class="tg-8d8j">2.05x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">alexnet </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">54.71%</td>
    <td class="tg-8d8j">54.79%</td>
    <td class="tg-8d8j">-0.15%</td>
    <td class="tg-8d8j">958.751</td>
    <td class="tg-8d8j">477.769</td>
    <td class="tg-8d8j">2.01x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">arcface</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">99.80%</td>
    <td class="tg-8d8j">99.80%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">225.096</td>
    <td class="tg-8d8j">126.563</td>
    <td class="tg-8d8j">1.78x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base mrpc dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">85.29%</td>
    <td class="tg-8d8j">86.03%</td>
    <td class="tg-8d8j">-0.86%</td>
    <td class="tg-8d8j">298.334</td>
    <td class="tg-8d8j">124.673</td>
    <td class="tg-8d8j">2.39x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert base mrpc static</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">85.54%</td>
    <td class="tg-8d8j">86.03%</td>
    <td class="tg-8d8j">-0.57%</td>
    <td class="tg-8d8j">624.429</td>
    <td class="tg-8d8j">254.639</td>
    <td class="tg-8d8j">2.45x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">bert squad model zoo dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">80.43519</td>
    <td class="tg-8d8j">80.67171</td>
    <td class="tg-8d8j">-0.29%</td>
    <td class="tg-8d8j">97.807</td>
    <td class="tg-8d8j">52.745</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">caffenet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">56.21%</td>
    <td class="tg-8d8j">56.30%</td>
    <td class="tg-8d8j">-0.16%</td>
    <td class="tg-8d8j">1432.981</td>
    <td class="tg-8d8j">540.284</td>
    <td class="tg-8d8j">2.65x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">caffenet </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">56.25%</td>
    <td class="tg-8d8j">56.30%</td>
    <td class="tg-8d8j">-0.09%</td>
    <td class="tg-8d8j">1460.212</td>
    <td class="tg-8d8j">540.81</td>
    <td class="tg-8d8j">2.70x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">densenet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">60.53%</td>
    <td class="tg-8d8j">60.96%</td>
    <td class="tg-8d8j">-0.71%</td>
    <td class="tg-8d8j">357.409</td>
    <td class="tg-8d8j">265.218</td>
    <td class="tg-8d8j">1.35x</td>
  </tr>
  <tr>
    <td class="tg-baqh">distilbert base mrpc</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">85.54%</td>
    <td class="tg-8d8j">84.56%</td>
    <td class="tg-8d8j">1.16%</td>
    <td class="tg-8d8j">1365.718</td>
    <td class="tg-8d8j">477.62</td>
    <td class="tg-8d8j">2.86x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">distilbert base mrpc </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">84.56%</td>
    <td class="tg-8d8j">84.56%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">524.955</td>
    <td class="tg-8d8j">476.394</td>
    <td class="tg-8d8j">1.10x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">duc</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">81.62%</td>
    <td class="tg-8d8j">81.92%</td>
    <td class="tg-8d8j">-0.37%</td>
    <td class="tg-8d8j">5.655</td>
    <td class="tg-8d8j">2.819</td>
    <td class="tg-8d8j">2.01x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">efficientnet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">77.57%</td>
    <td class="tg-8d8j">77.70%</td>
    <td class="tg-8d8j">-0.17%</td>
    <td class="tg-8d8j">1211.095</td>
    <td class="tg-8d8j">758.409</td>
    <td class="tg-8d8j">1.60x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">efficientnet </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">77.61%</td>
    <td class="tg-8d8j">77.70%</td>
    <td class="tg-8d8j">-0.12%</td>
    <td class="tg-8d8j">856.644</td>
    <td class="tg-8d8j">762.482</td>
    <td class="tg-8d8j">1.12x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">emotion ferplus</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">8.00%</td>
    <td class="tg-8d8j">8.00%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">925.428</td>
    <td class="tg-8d8j">694.985</td>
    <td class="tg-8d8j">1.33x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">34.09%</td>
    <td class="tg-8d8j">34.37%</td>
    <td class="tg-8d8j">-0.81%</td>
    <td class="tg-8d8j">13.819</td>
    <td class="tg-8d8j">5.889</td>
    <td class="tg-8d8j">2.35x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">faster rcnn </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">33.90%</td>
    <td class="tg-8d8j">34.37%</td>
    <td class="tg-8d8j">-1.37%</td>
    <td class="tg-8d8j">9.593</td>
    <td class="tg-8d8j">6.094</td>
    <td class="tg-8d8j">1.57x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">fcn</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">64.54%</td>
    <td class="tg-8d8j">64.98%</td>
    <td class="tg-8d8j">-0.68%</td>
    <td class="tg-8d8j">40.493</td>
    <td class="tg-8d8j">11.921</td>
    <td class="tg-8d8j">3.40x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">fcn </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">64.40%</td>
    <td class="tg-8d8j">64.98%</td>
    <td class="tg-8d8j">-0.89%</td>
    <td class="tg-8d8j">26.87</td>
    <td class="tg-8d8j">11.919</td>
    <td class="tg-8d8j">2.25x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">googlenet-12</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">67.71%</td>
    <td class="tg-8d8j">67.79%</td>
    <td class="tg-8d8j">-0.12%</td>
    <td class="tg-8d8j">771.387</td>
    <td class="tg-8d8j">571.351</td>
    <td class="tg-8d8j">1.35x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">googlenet-12 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">67.73%</td>
    <td class="tg-8d8j">67.79%</td>
    <td class="tg-8d8j">-0.09%</td>
    <td class="tg-8d8j">763.79</td>
    <td class="tg-8d8j">579.946</td>
    <td class="tg-8d8j">1.32x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf albert-base-v2 dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">91.40%</td>
    <td class="tg-8d8j">92.32%</td>
    <td class="tg-8d8j">-1.00%</td>
    <td class="tg-8d8j">156.958</td>
    <td class="tg-8d8j">105.894</td>
    <td class="tg-8d8j">1.48x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf bert-base-multilingual-cased dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">88.7022</td>
    <td class="tg-8d8j">89.1263</td>
    <td class="tg-8d8j">-0.48%</td>
    <td class="tg-8d8j">47.676</td>
    <td class="tg-8d8j">23.952</td>
    <td class="tg-8d8j">1.99x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf bert-base-uncased dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">89.58%</td>
    <td class="tg-8d8j">90.42%</td>
    <td class="tg-8d8j">-0.93%</td>
    <td class="tg-8d8j">199.365</td>
    <td class="tg-8d8j">104.847</td>
    <td class="tg-8d8j">1.90x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf camembert-base dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">88.47%</td>
    <td class="tg-8d8j">89.28%</td>
    <td class="tg-8d8j">-0.91%</td>
    <td class="tg-8d8j">182.604</td>
    <td class="tg-8d8j">105.447</td>
    <td class="tg-8d8j">1.73x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf distilbert-base-uncased&nbsp;&nbsp;&nbsp;dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">90.37%</td>
    <td class="tg-8d8j">91.06%</td>
    <td class="tg-8d8j">-0.76%</td>
    <td class="tg-8d8j">449.707</td>
    <td class="tg-8d8j">164.208</td>
    <td class="tg-8d8j">2.74x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf minilm-l12-h384-uncased&nbsp;&nbsp;&nbsp;dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">91.07%</td>
    <td class="tg-8d8j">90.97%</td>
    <td class="tg-8d8j">0.11%</td>
    <td class="tg-8d8j">466.585</td>
    <td class="tg-8d8j">247.708</td>
    <td class="tg-8d8j">1.88x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf minilm-l6-h384-uncased&nbsp;&nbsp;&nbsp;dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">89.91%</td>
    <td class="tg-8d8j">90.14%</td>
    <td class="tg-8d8j">-0.26%</td>
    <td class="tg-8d8j">523.59</td>
    <td class="tg-8d8j">354.049</td>
    <td class="tg-8d8j">1.48x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf roberta-base dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">90.85%</td>
    <td class="tg-8d8j">91.38%</td>
    <td class="tg-8d8j">-0.58%</td>
    <td class="tg-8d8j">183.586</td>
    <td class="tg-8d8j">107.7</td>
    <td class="tg-8d8j">1.70x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf spanbert dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">91.3983</td>
    <td class="tg-8d8j">91.9755</td>
    <td class="tg-8d8j">-0.63%</td>
    <td class="tg-8d8j">48.357</td>
    <td class="tg-8d8j">24.027</td>
    <td class="tg-8d8j">2.01x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">hf xlm-roberta-base dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">89.45%</td>
    <td class="tg-8d8j">90.10%</td>
    <td class="tg-8d8j">-0.72%</td>
    <td class="tg-8d8j">208.155</td>
    <td class="tg-8d8j">64.604</td>
    <td class="tg-8d8j">3.22x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v1</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">67.21%</td>
    <td class="tg-8d8j">67.24%</td>
    <td class="tg-8d8j">-0.04%</td>
    <td class="tg-8d8j">795.376</td>
    <td class="tg-8d8j">600.027</td>
    <td class="tg-8d8j">1.33x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">inception v1 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">67.21%</td>
    <td class="tg-8d8j">67.24%</td>
    <td class="tg-8d8j">-0.04%</td>
    <td class="tg-8d8j">780.704</td>
    <td class="tg-8d8j">591.812</td>
    <td class="tg-8d8j">1.32x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mask rcnn</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">33.13%</td>
    <td class="tg-8d8j">33.72%</td>
    <td class="tg-8d8j">-1.75%</td>
    <td class="tg-8d8j">11.61</td>
    <td class="tg-8d8j">5.577</td>
    <td class="tg-8d8j">2.08x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mask rcnn </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">33.28%</td>
    <td class="tg-8d8j">33.72%</td>
    <td class="tg-8d8j">-1.30%</td>
    <td class="tg-8d8j">8.639</td>
    <td class="tg-8d8j">5.534</td>
    <td class="tg-8d8j">1.56x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilebert mrpc</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">86.27%</td>
    <td class="tg-8d8j">86.27%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">591.94</td>
    <td class="tg-8d8j">515.485</td>
    <td class="tg-8d8j">1.15x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilebert squad mlperf&nbsp;&nbsp;&nbsp;dynamic</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">89.82276</td>
    <td class="tg-8d8j">90.0265</td>
    <td class="tg-8d8j">-0.23%</td>
    <td class="tg-8d8j">85.658</td>
    <td class="tg-8d8j">74.123</td>
    <td class="tg-8d8j">1.16x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">65.59%</td>
    <td class="tg-8d8j">66.89%</td>
    <td class="tg-8d8j">-1.94%</td>
    <td class="tg-8d8j">2370.927</td>
    <td class="tg-8d8j">1526.329</td>
    <td class="tg-8d8j">1.55x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">65.82%</td>
    <td class="tg-8d8j">66.89%</td>
    <td class="tg-8d8j">-1.60%</td>
    <td class="tg-8d8j">2216.018</td>
    <td class="tg-8d8j">1506.851</td>
    <td class="tg-8d8j">1.47x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v3 mlperf</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">75.58%</td>
    <td class="tg-8d8j">75.74%</td>
    <td class="tg-8d8j">-0.21%</td>
    <td class="tg-8d8j">2078.849</td>
    <td class="tg-8d8j">1028.313</td>
    <td class="tg-8d8j">2.02x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v3 mlperf </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">75.57%</td>
    <td class="tg-8d8j">75.74%</td>
    <td class="tg-8d8j">-0.22%</td>
    <td class="tg-8d8j">1762.617</td>
    <td class="tg-8d8j">999.313</td>
    <td class="tg-8d8j">1.76x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenetv2-12</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">68.38%</td>
    <td class="tg-8d8j">69.48%</td>
    <td class="tg-8d8j">-1.58%</td>
    <td class="tg-8d8j">2615.52</td>
    <td class="tg-8d8j">1645.083</td>
    <td class="tg-8d8j">1.59x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenetv2-12 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">68.51%</td>
    <td class="tg-8d8j">69.48%</td>
    <td class="tg-8d8j">-1.40%</td>
    <td class="tg-8d8j">2461.246</td>
    <td class="tg-8d8j">1674.355</td>
    <td class="tg-8d8j">1.47x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet v1 5 mlperf</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">76.15%</td>
    <td class="tg-8d8j">76.46%</td>
    <td class="tg-8d8j">-0.41%</td>
    <td class="tg-8d8j">766.332</td>
    <td class="tg-8d8j">431.923</td>
    <td class="tg-8d8j">1.77x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet v1 5 mlperf </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">76.14%</td>
    <td class="tg-8d8j">76.46%</td>
    <td class="tg-8d8j">-0.42%</td>
    <td class="tg-8d8j">575.336</td>
    <td class="tg-8d8j">430.825</td>
    <td class="tg-8d8j">1.34x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50 v1 5</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">72.26%</td>
    <td class="tg-8d8j">72.29%</td>
    <td class="tg-8d8j">-0.04%</td>
    <td class="tg-8d8j">747.307</td>
    <td class="tg-8d8j">431.092</td>
    <td class="tg-8d8j">1.73x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50 v1 5 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">72.20%</td>
    <td class="tg-8d8j">72.29%</td>
    <td class="tg-8d8j">-0.12%</td>
    <td class="tg-8d8j">564.212</td>
    <td class="tg-8d8j">431.495</td>
    <td class="tg-8d8j">1.31x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50-v1-12</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">74.81%</td>
    <td class="tg-8d8j">74.99%</td>
    <td class="tg-8d8j">-0.24%</td>
    <td class="tg-8d8j">594.291</td>
    <td class="tg-8d8j">449.209</td>
    <td class="tg-8d8j">1.32x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet50-v1-12 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">74.76%</td>
    <td class="tg-8d8j">74.99%</td>
    <td class="tg-8d8j">-0.31%</td>
    <td class="tg-8d8j">590.513</td>
    <td class="tg-8d8j">449.934</td>
    <td class="tg-8d8j">1.31x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">roberta base mrpc</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">90.69%</td>
    <td class="tg-8d8j">89.95%</td>
    <td class="tg-8d8j">0.82%</td>
    <td class="tg-8d8j">643.025</td>
    <td class="tg-8d8j">253.041</td>
    <td class="tg-8d8j">2.54x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">shufflenet-v2-12</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">66.13%</td>
    <td class="tg-8d8j">66.36%</td>
    <td class="tg-8d8j">-0.35%</td>
    <td class="tg-8d8j">2354.511</td>
    <td class="tg-8d8j">1461.472</td>
    <td class="tg-8d8j">1.61x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">shufflenet-v2-12 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">66.12%</td>
    <td class="tg-8d8j">66.36%</td>
    <td class="tg-8d8j">-0.36%</td>
    <td class="tg-8d8j">1850.085</td>
    <td class="tg-8d8j">1368.347</td>
    <td class="tg-8d8j">1.35x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">squeezenet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">56.54%</td>
    <td class="tg-8d8j">56.87%</td>
    <td class="tg-8d8j">-0.58%</td>
    <td class="tg-8d8j">2484.357</td>
    <td class="tg-8d8j">1912.365</td>
    <td class="tg-8d8j">1.30x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">squeezenet </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">56.39%</td>
    <td class="tg-8d8j">56.87%</td>
    <td class="tg-8d8j">-0.83%</td>
    <td class="tg-8d8j">2526.016</td>
    <td class="tg-8d8j">1911.319</td>
    <td class="tg-8d8j">1.32x</td>
  </tr>
  <tr>
    <td class="tg-baqh">ssd mobilenet v1</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">22.44%</td>
    <td class="tg-8d8j">23.10%</td>
    <td class="tg-8d8j">-2.86%</td>
    <td class="tg-8d8j">710.173</td>
    <td class="tg-8d8j">549.548</td>
    <td class="tg-8d8j">1.29x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet v1 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">22.44%</td>
    <td class="tg-8d8j">23.10%</td>
    <td class="tg-8d8j">-2.86%</td>
    <td class="tg-8d8j">622.58</td>
    <td class="tg-8d8j">497.419</td>
    <td class="tg-8d8j">1.25x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet v1-2</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">22.96%</td>
    <td class="tg-8d8j">23.02%</td>
    <td class="tg-8d8j">-0.26%</td>
    <td class="tg-8d8j">652.141</td>
    <td class="tg-8d8j">507.766</td>
    <td class="tg-8d8j">1.28x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet v1-2 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">22.96%</td>
    <td class="tg-8d8j">23.02%</td>
    <td class="tg-8d8j">-0.26%</td>
    <td class="tg-8d8j">573.3</td>
    <td class="tg-8d8j">470.424</td>
    <td class="tg-8d8j">1.22x</td>
  </tr>
  <tr>
    <td class="tg-baqh">ssd mobilenet v2</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">24.03%</td>
    <td class="tg-8d8j">24.67%</td>
    <td class="tg-8d8j">-2.59%</td>
    <td class="tg-8d8j">527.674</td>
    <td class="tg-8d8j">396.265</td>
    <td class="tg-8d8j">1.33x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd-12</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">18.92%</td>
    <td class="tg-8d8j">18.98%</td>
    <td class="tg-8d8j">-0.32%</td>
    <td class="tg-8d8j">31.241</td>
    <td class="tg-8d8j">8.766</td>
    <td class="tg-8d8j">3.56x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd-12 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">18.63%</td>
    <td class="tg-8d8j">18.98%</td>
    <td class="tg-8d8j">-1.84%</td>
    <td class="tg-8d8j">23.721</td>
    <td class="tg-8d8j">8.866</td>
    <td class="tg-8d8j">2.68x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">tiny yolov3</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">11.82%</td>
    <td class="tg-8d8j">12.42%</td>
    <td class="tg-8d8j">-4.83%</td>
    <td class="tg-8d8j">647.166</td>
    <td class="tg-8d8j">514.415</td>
    <td class="tg-8d8j">1.26x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ultraface</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">83.34%</td>
    <td class="tg-8d8j">83.65%</td>
    <td class="tg-8d8j">-0.37%</td>
    <td class="tg-8d8j">314.504</td>
    <td class="tg-8d8j">125.561</td>
    <td class="tg-8d8j">2.50x</td>
  </tr>
  <tr>
    <td class="tg-baqh">vgg16</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">66.67%</td>
    <td class="tg-8d8j">66.69%</td>
    <td class="tg-8d8j">-0.03%</td>
    <td class="tg-8d8j">221.615</td>
    <td class="tg-8d8j">98.201</td>
    <td class="tg-8d8j">2.26x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">vgg16 </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">66.69%</td>
    <td class="tg-8d8j">66.69%</td>
    <td class="tg-8d8j">0.00%</td>
    <td class="tg-8d8j">304.094</td>
    <td class="tg-8d8j">98.329</td>
    <td class="tg-8d8j">3.09x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">vgg16 model zoo</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">72.32%</td>
    <td class="tg-8d8j">72.40%</td>
    <td class="tg-8d8j">-0.11%</td>
    <td class="tg-8d8j">316.543</td>
    <td class="tg-8d8j">98.489</td>
    <td class="tg-8d8j">3.21x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">vgg16 model zoo </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">72.31%</td>
    <td class="tg-8d8j">72.40%</td>
    <td class="tg-8d8j">-0.12%</td>
    <td class="tg-8d8j">315.612</td>
    <td class="tg-8d8j">98.46</td>
    <td class="tg-8d8j">3.21x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">yolov3</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">26.92%</td>
    <td class="tg-8d8j">28.73%</td>
    <td class="tg-8d8j">-6.30%</td>
    <td class="tg-8d8j">119.633</td>
    <td class="tg-8d8j">53.371</td>
    <td class="tg-8d8j">2.24x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">yolov4</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">32.33%</td>
    <td class="tg-8d8j">33.71%</td>
    <td class="tg-8d8j">-4.09%</td>
    <td class="tg-8d8j">49.299</td>
    <td class="tg-8d8j">32.879</td>
    <td class="tg-8d8j">1.50x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">zfnet</td>
    <td class="tg-8d8j">qLinear</td>
    <td class="tg-8d8j">55.84%</td>
    <td class="tg-8d8j">55.96%</td>
    <td class="tg-8d8j">-0.21%</td>
    <td class="tg-8d8j">462.281</td>
    <td class="tg-8d8j">268.316</td>
    <td class="tg-8d8j">1.72x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">zfnet </td>
    <td class="tg-8d8j">qdq</td>
    <td class="tg-8d8j">55.86%</td>
    <td class="tg-8d8j">55.96%</td>
    <td class="tg-8d8j">-0.18%</td>
    <td class="tg-8d8j">465.44</td>
    <td class="tg-8d8j">265.581</td>
    <td class="tg-8d8j">1.75x</td>
  </tr>
</tbody>
</table>

### MXNet Models with MXNet 1.9.1

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-wa1i{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8d8j{text-align:center;vertical-align:bottom}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" rowspan="2">model</th>
    <th class="tg-wa1i" colspan="3">Accuracy</th>
    <th class="tg-wa1i" colspan="3">Performance/1s4c8ins1bs/throughput(samples/sec)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Acc&nbsp;&nbsp;&nbsp;Ratio<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</th>
    <th class="tg-wa1i">INT8</th>
    <th class="tg-wa1i">FP32</th>
    <th class="tg-wa1i">Performance&nbsp;&nbsp;&nbsp;Gain<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8d8j">inception&nbsp;&nbsp;&nbsp;v3</td>
    <td class="tg-8d8j">77.65%</td>
    <td class="tg-8d8j">0.16%</td>
    <td class="tg-8d8j">94.236</td>
    <td class="tg-8d8j">58.046</td>
    <td class="tg-8d8j">1.62x</td>
    <td class="tg-8d8j">2.05x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet 1.0</td>
    <td class="tg-8d8j">72.23%</td>
    <td class="tg-8d8j">-0.86%</td>
    <td class="tg-8d8j">436.461</td>
    <td class="tg-8d8j">314.811</td>
    <td class="tg-8d8j">1.39x</td>
    <td class="tg-8d8j">2.01x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">mobilenet v2 1.0</td>
    <td class="tg-8d8j">70.87%</td>
    <td class="tg-8d8j">-0.16%</td>
    <td class="tg-8d8j">270.779</td>
    <td class="tg-8d8j">229.21</td>
    <td class="tg-8d8j">1.18x</td>
    <td class="tg-8d8j">1.78x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 152 v1</td>
    <td class="tg-8d8j">78.54%</td>
    <td class="tg-8d8j">-0.30%</td>
    <td class="tg-8d8j">66.616</td>
    <td class="tg-8d8j">36.553</td>
    <td class="tg-8d8j">1.82x</td>
    <td class="tg-8d8j">2.39x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 18 v1</td>
    <td class="tg-8d8j">70.14%</td>
    <td class="tg-8d8j">-0.19%</td>
    <td class="tg-8d8j">429.858</td>
    <td class="tg-8d8j">224.103</td>
    <td class="tg-8d8j">1.92x</td>
    <td class="tg-8d8j">2.45x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">resnet 50 v1</td>
    <td class="tg-8d8j">76.33%</td>
    <td class="tg-8d8j">-0.50%</td>
    <td class="tg-8d8j">182.561</td>
    <td class="tg-8d8j">94.152</td>
    <td class="tg-8d8j">1.94x</td>
    <td class="tg-8d8j">1.85x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">squeezenet 1.0</td>
    <td class="tg-8d8j">56.97%</td>
    <td class="tg-8d8j">-0.26%</td>
    <td class="tg-8d8j">331.716</td>
    <td class="tg-8d8j">242.763</td>
    <td class="tg-8d8j">1.37x</td>
    <td class="tg-8d8j">2.65x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd mobilenet 1.0</td>
    <td class="tg-8d8j">75.54%</td>
    <td class="tg-8d8j">-0.79%</td>
    <td class="tg-8d8j">53.659</td>
    <td class="tg-8d8j">27.161</td>
    <td class="tg-8d8j">1.98x</td>
    <td class="tg-8d8j">2.70x</td>
  </tr>
  <tr>
    <td class="tg-8d8j">ssd resnet50 v1</td>
    <td class="tg-8d8j">80.23%</td>
    <td class="tg-8d8j">-0.05%</td>
    <td class="tg-8d8j">37.629</td>
    <td class="tg-8d8j">16.798</td>
    <td class="tg-8d8j">2.24x</td>
    <td class="tg-8d8j">1.35x</td>
  </tr>
</tbody>
</table>

## Validated Pruning Examples

<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Task</br>Dataset</th>
    <th rowspan="2">Dense Accuracy<br>Sparse Accuracy</th>
    <th rowspan="2">Relative Drop</th>
    <th rowspan="2">Sparsity ratio<br>Sparsity Pattern</th>
    <th rowspan="2">Comments<br>Balanced or unbalanced ratio</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet18</td>
    <td>image classification</br>ImageNet</td>
    <td>top-1% acc = 69.76</br>top-1% acc = 69.47</td>
    <td>-0.42%</td>    
    <td>30%</td>
    <td>magnitude</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>image classification</br>ImageNet</td>
    <td>top-1% acc = 76.13</br>top-1% acc = 76.11</td>
    <td>-0.03%</td>    
    <td>30%</td>
    <td>magnitude</td>
  </tr> 
  <tr>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>image classification</br>ImageNet</td>
    <td>top-1% acc = 76.13</br>top-1% acc = 76.01</td>
    <td>-0.16%</td>    
    <td>30%</td>
    <td>magnitude</br>Post Training Quantization</td>    
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>ResNet50</td>
    <td>image classification</br>ImageNet</td>
    <td>top-1% acc = 76.13</br>top-1% acc = 75.90</td>
    <td>-0.30%</td> 
    <td>30%</td>
    <td>magnitude</br>Quantization Aware Training</td>   
  </tr> 
  <tr>
  </tr>     
  <tr>
    <td>Bert-Large</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=91.34</br>f1=90.7</td>
    <td>-0.07%</td>
    <td>80%</br>structured 2x1</td>
    <td>group lasso</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>MNLI</td>
    <td>[m, mm] = [84.57, 84.79]</br>[m, mm] = [82.45, 83.27]</td>
    <td>[-2.51%, -1.80%]</td>
    <td>70%</br>unstructured</td>
    <td>Prune once for all</br>balanced</td>    
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>MNLI</td>
    <td>[m, mm] = [84.57, 84.79]</br>[m, mm] = [83.20, 84.11]</td>
    <td>[-1.62%, -0.80%]</td>
    <td>50%</br>structured 1:2</td>
    <td>Prune once for all</br>balanced</td>    
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy = 92.32</br>accuracy = 91.51</td>
    <td>-0.88%</td>
    <td>70%</br>unstructured</td>
    <td>Prune once for all</br>balanced</td>    
  </tr>
  <tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy = 92.32</br>accuracy = 92.20</td>
    <td>-0.13%</td>
    <td>50%</br>structured 1:2</td>
    <td>Prune once for all</br>balanced</td>       
  </tr>
  <tr>  
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy = 92.32</br>accuracy = 91.97</td>
    <td>-0.38%</td>
    <td>20%</br>unstructured</td>
    <td>gradient sensitivity</br>balanced</td>       
  </tr>
  <tr>  
  </tr>  
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>QQP</td>
    <td>[accuracy, f1] = [91.10, 88.05]</br>[accuracy, f1] = [90.48, 87.06]</td>
    <td>[-0.68%, -1.12%]</td>
    <td>70%</br>unstructured</td>
    <td>Prune once for all</br>balanced</td>        
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>QQP</td>
    <td>[accuracy, f1] = [91.10, 88.05]</br>[accuracy, f1] = [90.92, 87.78]</td>
    <td>[-0.20%, -0.31%]</td>
    <td>50%</br>structured 1:2</td>
    <td>Prune once for all</br>balanced</td>        
  </tr>
  <tr>
  </tr>   
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>QNLI</td>
    <td>accuracy = 91.54</br>accuracy = 90.39</td>
    <td>-1.26%</td>
    <td>70%</br>unstructured</td>
    <td>Prune once for all</br>balanced</td>        
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>text classification</br>QNLI</td>
    <td>accuracy = 91.54</br>accuracy = 90.87</td>
    <td>-0.73%</td>
    <td>50%</br>structured 1:2</td>
    <td>Prune once for all</br>balanced</td>      
  </tr>
  <tr>
  </tr>   
  <tr>
    <td>Bert-Base</td>
    <td>question answering</td>
    <td>[em, f1] = [79.34, 87.10]</br>[em, f1] = [77.27, 85.75]</td>
    <td>[-2.61%, -1.54%]</td>
    <td>70%</br>unstructured</td>
    <td>Prune once for all</br>balanced</td>   
  </tr>  
  <tr>
  </tr>
  <tr>
    <td>Bert-Base</td>
    <td>question answering</td>
    <td>[em, f1] = [79.34, 87.10]</br>[em, f1] = [78.03, 86.50]</td>
    <td>[-1.65%, -0.69%]</td>
    <td>50%</br>structured 1:2</td>
    <td>Prune once for all</br>balanced</td>       
  </tr>  
  <tr>
  </tr>     
  <tr>
    <td>Bert-Mini</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1]=76.87</br>f1=76.2</td>
    <td>-0.80%</td>
    <td>80%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=76.87</br>f1=77.62</td>
    <td>+0.98%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1]=86.90</br>f1=86.15</td>
    <td>-0.86%</td>
    <td>80%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=86.90</br>f1=87.50</td>
    <td>+0.69%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-base-uncased</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1]=88.59</br>f1=87.78</td>
    <td>-0.92%</td>
    <td>80%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-base-uncased</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=88.59</br>f1=89.40</td>
    <td>+0.91%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-large</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1]=91.23</br>f1=90.91</td>
    <td>-0.35%</td>
    <td>80%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-large</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=91.23</br>f1=91.67</td>
    <td>+0.48%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>MRPC</td>
    <td>f1=87.52</br>f1=87.22</td>
    <td>-0.34%</td>
    <td>90%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>MRPC</td>
    <td>f1=87.52</br>f1=87.33</td>
    <td>-0.22%</td>
    <td>90%</br>structured 4x1</td>
    <td>snip momentum</br>balanced</td>  
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>MRPC</td>
    <td>f1=87.52</br>f1=86.89</td>
    <td>-0.72%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>MRPC</td>
    <td>f1=87.52</br>f1=86.8</td>
    <td>-0.83%</td>
    <td>60%</br>structured per channel</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>text classification</br>MRPC</td>
    <td>f1=90.26</br>f1=89.85</td>
    <td>-0.46%</td>
    <td>90%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>  
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>Distilbert-base-uncased</td>
    <td>text classification</br>MRPC</td>
    <td>f1=90.26</br>f1=90.88</td>
    <td>+0.69%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy=87.61</br>accuracy=86.92</td>
    <td>-0.79%</td>
    <td>90%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>  
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy=87.61</br>accuracy=87.73</td>
    <td>+0.14%</td>
    <td>50%</br>structured 2:4</td>
    <td>snip momentum</br>balanced</td>
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>Bert-Mini</td>
    <td>text classification</br>SST-2</td>
    <td>accuracy=87.61</br>accuracy=86.92</td>
    <td>-0.79%</td>
    <td>50%</br>structured per channel</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr> 
</tbody>
</table>

## Validated Knowledge Distillation Examples
|  Example Name       | Dataset   | Student<br>(Metrics)                 | Teacher<br>(Metrics)               | Student With Distillation<br>(Metrics Improvement)  | Student With Distributed Distillation<br>(Metrics Improvement)  |
|---------------------|-----------|--------------------------------------|------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| MobileNet example   | CIFAR-10  | MobileNetV2-0.35<br>(0.7965 ACC)     | WideResNet40-2<br>(0.9522 ACC)     |   0.8178 ACC<br>(0.0213 ACC)                        |   0.8235 ACC<br>(0.027 ACC)                        |
| CNN example         | CIFAR-100 | CNN-2<br>(0.5494 ACC)                | CNN-10<br>(0.7153 ACC)             |   0.5540 ACC<br>(0.0046 ACC)                        |   0.5523 ACC<br>(0.0029 ACC)                        |
| VGG example         | CIFAR-100 | VGG-8-BN<br>(0.7022 ACC)             | VGG-13-BN<br>(0.7415 ACC)          |   0.7025 ACC<br>(0.0003 ACC)                        |   WIP                        |
| ResNet example      | ImageNet  | ResNet18<br>(0.6739 ACC)             | ResNet50<br>(0.7399 ACC)           |   0.6845 ACC<br>(0.0106 ACC)                        |   WIP                        |
| BlendCnn example    |   MRPC    | BlendCnn<br>(0.7034 ACC)             | BERT-Base<br>(0.8382 ACC)          |   0.7034 ACC<br>(0 ACC)                             |   WIP                        |
| BiLSTM example      |  SST-2    | BiLSTM<br>(0.8314 ACC)               | RoBERTa-Base<br>(0.9403 ACC)       |   0.9048 ACC<br>(0.0734 ACC)                        |   WIP                        |
|DistilBERT example   |  SQuAD    | DistilBERT<br>(0.7323/0.8256 EM/F1)  | BERT-Base<br>(0.8084/0.8814 EM/F1) |   0.7442/0.8371 EM/F1<br>(0.0119/0.0115 EM/F1)      |   WIP                        |
|TinyBERT example     |  MNLI     | TinyBERT<br>(0.8018/0.8044 m/mm)     | BERT-Base<br>(0.8363/0.8411 m/mm)  |   0.8025/0.8074 m/mm<br>(0.0007/0.0030 m/mm)        |   WIP                        |
|BERT-3 example       |  QQP      | BERT-3<br>(0.8626/0.8213 EM/F1)      | BERT-Base<br>(0.9091/0.8782 EM/F1) |   0.8684/0.8259 EM/F1<br>(0.0058/0.0046 EM/F1)      |   WIP                        |
|DistilRoBERTa example|  COLA     | DistilRoBERTa<br>(0.6057 ACC)        | RoBERTa-Large<br>(0.6455 ACC)      |   0.6187 ACC<br>(0.0130 ACC)                        |   WIP                        |

## Validated ONNX QDQ INT8 Models on Multiple Hardware through ONNX Runtime

<table class="tg">
<thead>
  <tr>
    <th class="tg-y3we">Model (ONNX QDQ)</th>
    <th class="tg-pm1l">AWS c6i.2xlarge (Intel)<br>CPU Execution Provider</th>
    <th class="tg-pm1l">AWS c6a.2xlarge (AMD)<br>CPU Execution Provider</th>
    <th class="tg-pm1l">AWS c6g.2xlarge (ARM)<br>CPU Execution Provider</th>
    <th class="tg-8d8j">NVidia A100<br>CUDA Execution Provider</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-cwad">ResNet50</td>
    <td class="tg-pm1l">74.76%</td>
    <td class="tg-pm1l">68.95%</td>
    <td class="tg-pm1l">74.76%</td>
    <td class="tg-6q5x">74.75%</td>
  </tr>
  <tr>
    <td class="tg-cwad">BERT-base</td>
    <td class="tg-pm1l">85.54%</td>
    <td class="tg-pm1l">84.56%</td>
    <td class="tg-pm1l">85.54%</td>
    <td class="tg-6q5x">84.31%</td>
  </tr>
  <tr>
    <td class="tg-cwad">ResNet50 V1.5</td>
    <td class="tg-pm1l">72.20%</td>
    <td class="tg-pm1l">67.70%</td>
    <td class="tg-pm1l">72.20%</td>
    <td class="tg-6q5x">72.29%</td>
  </tr>
  <tr>
    <td class="tg-cwad">MobileNet V2</td>
    <td class="tg-pm1l">65.82%</td>
    <td class="tg-pm1l">58.56%</td>
    <td class="tg-pm1l">65.83%</td>
    <td class="tg-pm1l">65.63%</td>
  </tr>
  <tr>
    <td class="tg-cwad">SSD MobileNet V1</td>
    <td class="tg-pm1l">22.45%</td>
    <td class="tg-pm1l">16.53%</td>
    <td class="tg-pm1l">22.45%</td>
    <td class="tg-pm1l">22.35%</td>
  </tr>
  <tr>
    <td class="tg-cwad">DistilBERT base MRPC</td>
    <td class="tg-pm1l">84.56%</td>
    <td class="tg-pm1l">83.82%</td>
    <td class="tg-pm1l">84.56%</td>
    <td class="tg-6q5x">84.56%</td>
  </tr>
  <tr>
    <td class="tg-cwad">SqueezeNet</td>
    <td class="tg-pm1l">56.54%</td>
    <td class="tg-pm1l">53.52%</td>
    <td class="tg-pm1l">56.54%</td>
    <td class="tg-6q5x">56.55%</td>
  </tr>
  <tr>
    <td class="tg-cwad">SSD</td>
    <td class="tg-pm1l">18.63%</td>
    <td class="tg-pm1l">18.54%</td>
    <td class="tg-pm1l">18.63%</td>
    <td class="tg-6q5x">18.61%</td>
  </tr>
  <tr>
    <td class="tg-cwad">AlexNet</td>
    <td class="tg-pm1l">54.71%</td>
    <td class="tg-pm1l">47.06%</td>
    <td class="tg-pm1l">54.71%</td>
    <td class="tg-pm1l">54.79%</td>
  </tr>
  <tr>
    <td class="tg-cwad">CaffeNet</td>
    <td class="tg-pm1l">56.25%</td>
    <td class="tg-pm1l">52.35%</td>
    <td class="tg-pm1l">56.27%</td>
    <td class="tg-pm1l">56.24%</td>
  </tr>
  <tr>
    <td class="tg-cwad">GoogleNet</td>
    <td class="tg-pm1l">67.73%</td>
    <td class="tg-pm1l">63.56%</td>
    <td class="tg-pm1l">67.72%</td>
    <td class="tg-6q5x">67.76%</td>
  </tr>
  <tr>
    <td class="tg-cwad">ZFNet</td>
    <td class="tg-pm1l">55.86%</td>
    <td class="tg-pm1l">45.09%</td>
    <td class="tg-pm1l">55.86%</td>
    <td class="tg-pm1l">55.89%</td>
  </tr>
  <tr>
    <td class="tg-cwad">Inception V1</td>
    <td class="tg-pm1l">67.21%</td>
    <td class="tg-pm1l">63.03%</td>
    <td class="tg-pm1l">67.20%</td>
    <td class="tg-6q5x">67.21%</td>
  </tr>
  <tr>
    <td class="tg-cwad">SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td class="tg-pm1l">22.86%</td>
    <td class="tg-pm1l">16.94%</td>
    <td class="tg-pm1l">22.80%</td>
    <td class="tg-pm1l">22.87%</td>
  </tr>
  <tr>
    <td class="tg-cwad">Mobile bert MRPC</td>
    <td class="tg-pm1l">85.54%</td>
    <td class="tg-pm1l">84.56%</td>
    <td class="tg-pm1l">85.54%</td>
    <td class="tg-pm1l">85.54%</td>
  </tr>
  <tr>
    <td class="tg-cwad">Roberta base MRPC</td>
    <td class="tg-pm1l">89.46%</td>
    <td class="tg-pm1l">90.44%</td>
    <td class="tg-pm1l">89.71%</td>
    <td class="tg-pm1l">89.71%</td>
  </tr>
  <tr>
    <td class="tg-cwad">ResNet50 V1.5 MLPerf</td>
    <td class="tg-pm1l">76.14%</td>
    <td class="tg-pm1l">72.80%</td>
    <td class="tg-pm1l">76.14%</td>
    <td class="tg-6q5x">76.17%</td>
  </tr>
  <tr>
    <td class="tg-cwad">VGG16</td>
    <td class="tg-pm1l">66.69%</td>
    <td class="tg-pm1l">64.25%</td>
    <td class="tg-pm1l">66.69%</td>
    <td class="tg-pm1l">66.64%</td>
  </tr>
  <tr>
    <td class="tg-cwad">VGG16 (ONNX Model Zoo)</td>
    <td class="tg-pm1l">72.31%</td>
    <td class="tg-pm1l">69.35%</td>
    <td class="tg-pm1l">72.32%</td>
    <td class="tg-pm1l">72.34%</td>
  </tr>
  <tr>
    <td class="tg-cwad">MobileNet V3 MLPerf</td>
    <td class="tg-pm1l">75.57%</td>
    <td class="tg-pm1l">70.78%</td>
    <td class="tg-pm1l">75.56%</td>
    <td class="tg-6q5x">75.52%</td>
  </tr>
  <tr>
    <td class="tg-cwad">EfficientNet</td>
    <td class="tg-pm1l">77.61%</td>
    <td class="tg-pm1l">76.52%</td>
    <td class="tg-pm1l">77.56%</td>
    <td class="tg-pm1l">77.60%</td>
  </tr>
  <tr>
    <td class="tg-cwad">MobileNet V2 (ONNX Model Zoo)</td>
    <td class="tg-pm1l">68.51%</td>
    <td class="tg-pm1l">62.48%</td>
    <td class="tg-pm1l">68.58%</td>
    <td class="tg-pm1l">68.48%</td>
  </tr>
  <tr>
    <td class="tg-413a">ShuffleNet V2</td>
    <td class="tg-pm1l">66.12%</td>
    <td class="tg-pm1l">58.41%</td>
    <td class="tg-pm1l">66.11%</td>
    <td class="tg-pm1l">66.11%</td>
  </tr>
</tbody>
</table>


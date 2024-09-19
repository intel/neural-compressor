# Validated Models

Intel® Neural Compressor validated examples with multiple compression techniques. The typical examples link can be found in [example tables](https://github.com/intel/neural-compressor/blob/master/examples/README.md), and the performance/accuracy results is available here.

1. [Validated Quantization Examples](#Validated-Quantization-Examples)

   1.1. [TensorFlow Models with TensorFlow 2.16.1](#tensorflow-models-with-tensorflow-2161)

   1.2. [Keras Models with keras 2.15.1](#keras-models-with-keras-2151)

   1.2. [PyTorch Models with Torch 2.3.0+cpu in PTQ Mode](#pytorch-models-with-torch-230cpu-in-ptq-mode)

   1.3. [PyTorch Models with Torch 2.3.0+cpu in QAT Mode](#pytorch-models-with-torch-230cpu-in-qat-mode)

   1.4. [ONNX Models with ONNX Runtime 1.18.1](#onnx-models-with-onnx-runtime-1181)

2. [Validated Pruning Examples](#Validated-Pruning-Examples)

3. [Validated Knowledge Distillation Examples](#Validated-Knowledge-Distillation-Examples)

4. [Validated ONNX QDQ INT8 Models on Multiple Hardware through ONNX Runtime](#validated-onnx-qdq-int8-models-on-multiple-hardware-through-onnx-runtime)

## Validated Quantization Examples

System summary: Test by Intel on 7/22/2024. 1-node, 1x Intel(R) Xeon(R) Platinum 8480+ @3.8GHz, 56 cores/socket, HT On, Turbo On, Total Memory 512GB (16x32GB DDR5 4800 MT/s [4800 MT/s]), BIOS EGSDCRB1.SYS.0081.D18.2205301336, microcode 0x2b000590,  
Ubuntu 24.04 LTS, gcc (GCC) 13.2.0 (Ubuntu 13.2.0-23ubuntu4), DL Models, Frameworks: TensorFlow/ONNXRT/PyTorch, Datatype: FP32/INT8/BF16.  
Using 1 socket, 4 cores/instance, 14 instances and batch size 1 to benchmark most of the model.

Performance varies by use, configuration and other factors.  
For more complete information about performance and benchmark results, visit www.intel.com/benchmarks

### TensorFlow Models with TensorFlow 2.16.1

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Example</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance 1s4c14ins1bs<br>Throughput(samples/sec)</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio<br>[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio<br>[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50 v1.0</td>
    <td>pb</td>
    <td>0.74112</td>
    <td>0.74274</td>
    <td>-0.00218</td>
    <td>1732.923</td>
    <td>578.879</td>
    <td>2.993584</td>
  </tr>
  <tr>
    <td>ResNet50 v1.5</td>
    <td>pb</td>
    <td>0.7625</td>
    <td>0.76464</td>
    <td>-0.0028</td>
    <td>1535.199</td>
    <td>529.999</td>
    <td>2.896607</td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>pb</td>
    <td>0.77524</td>
    <td>0.76448</td>
    <td>0.014075</td>
    <td>1048.361</td>
    <td>384.018</td>
    <td>2.729979</td>
  </tr>
  <tr>
    <td>Inception V1</td>
    <td>pb</td>
    <td>0.70454</td>
    <td>0.69738</td>
    <td>0.010267</td>
    <td>2079.239</td>
    <td>927.817</td>
    <td>2.241001</td>
  </tr>
  <tr>
    <td>Inception V2</td>
    <td>pb</td>
    <td>0.74326</td>
    <td>0.73966</td>
    <td>0.004867</td>
    <td>1644.364</td>
    <td>840.531</td>
    <td>1.95634</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>pb</td>
    <td>0.7672</td>
    <td>0.76746</td>
    <td>-0.00034</td>
    <td>1076.103</td>
    <td>401.885</td>
    <td>2.677639</td>
  </tr>
  <tr>
    <td>Inception V4</td>
    <td>pb</td>
    <td>0.80126</td>
    <td>0.80272</td>
    <td>-0.00182</td>
    <td>704.961</td>
    <td>199.28</td>
    <td>3.53754</td>
  </tr>
  <tr>
    <td>Inception ResNet V2</td>
    <td>pb</td>
    <td>0.80252</td>
    <td>0.80398</td>
    <td>-0.00182</td>
    <td>313.966</td>
    <td>178.274</td>
    <td>1.761143</td>
  </tr>
  <tr>
    <td>DenseNet-161</td>
    <td>pb</td>
    <td>0.76288</td>
    <td>0.76286</td>
    <td>2.62E-05</td>
    <td>279.204</td>
    <td>214.031</td>
    <td>1.304503</td>
  </tr>
  <tr>
    <td>MobileNet V1</td>
    <td>pb</td>
    <td>0.7179</td>
    <td>0.70956</td>
    <td>0.011754</td>
    <td>4199.131</td>
    <td>1506.681</td>
    <td>2.787007</td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>pb</td>
    <td>0.72484</td>
    <td>0.71756</td>
    <td>0.010145</td>
    <td>2170.387</td>
    <td>1445.045</td>
    <td>1.501951</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>pb</td>
    <td>0.72692</td>
    <td>0.70886</td>
    <td>0.025478</td>
    <td>1388.622</td>
    <td>203.393</td>
    <td>6.827285</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>pb</td>
    <td>0.72672</td>
    <td>0.71014</td>
    <td>0.023348</td>
    <td>1236.115</td>
    <td>169.736</td>
    <td>7.282574</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>0.6909</td>
    <td>0.69028</td>
    <td>0.000898</td>
    <td>411.79</td>
    <td>284.534</td>
    <td>1.447244</td>
  </tr>
  <tr>
    <td>ResNetV2 50</td>
    <td>pb</td>
    <td>0.70374</td>
    <td>0.69642</td>
    <td>0.010511</td>
    <td>779.416</td>
    <td>539.535</td>
    <td>1.444607</td>
  </tr>
  <tr>
    <td>ResNetV2 101</td>
    <td>pb</td>
    <td>0.72644</td>
    <td>0.7187</td>
    <td>0.010769</td>
    <td>492.002</td>
    <td>295.766</td>
    <td>1.663484</td>
  </tr>
  <tr>
    <td>ResNetV2 152</td>
    <td>pb</td>
    <td>0.7312</td>
    <td>0.72368</td>
    <td>0.010391</td>
    <td>348.385</td>
    <td>205.717</td>
    <td>1.693516</td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>pb</td>
    <td>0.81392</td>
    <td>0.8192</td>
    <td>-0.00645</td>
    <td>230.53</td>
    <td>132.657</td>
    <td>1.73779</td>
  </tr>
  <tr>
    <td>SSD ResNet50 V1</td>
    <td>pb</td>
    <td>0.3791</td>
    <td>0.38</td>
    <td>-0.00237</td>
    <td>135.711</td>
    <td>28.752</td>
    <td>4.720054</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1</td>
    <td>pb</td>
    <td>0.22995</td>
    <td>0.23127</td>
    <td>-0.00571</td>
    <td>1237.695</td>
    <td>719.299</td>
    <td>1.720696</td>
  </tr>
  <tr>
    <td>SSD ResNet50 v1</td>
    <td>ckpt</td>
    <td>0.37881</td>
    <td>0.38</td>
    <td>-0.00313</td>
    <td>130.537</td>
    <td>22.046</td>
    <td>5.921119</td>
  </tr>
  <tr>
    <td>SSD MobileNet v1</td>
    <td>ckpt</td>
    <td>0.22963</td>
    <td>0.23127</td>
    <td>-0.00709</td>
    <td>1234.562</td>
    <td>529.342</td>
    <td>2.332258</td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet101</td>
    <td>pb</td>
    <td>0.30319</td>
    <td>0.30387</td>
    <td>-0.00224</td>
    <td>144.211</td>
    <td>22.637</td>
    <td>6.370588</td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet50</td>
    <td>pb</td>
    <td>0.2661</td>
    <td>0.26586</td>
    <td>0.000903</td>
    <td>164.553</td>
    <td>28.376</td>
    <td>5.79902</td>
  </tr>
  <tr>
    <td>YOLOv3</td>
    <td>pb</td>
    <td>0.83279</td>
    <td>0.82353</td>
    <td>0.011244</td>
    <td>247.559</td>
    <td>81.447</td>
    <td>3.03951</td>
  </tr>
  <tr>
    <td>BERT large SQuAD</td>
    <td>pb</td>
    <td>92.44432</td>
    <td>92.98612</td>
    <td>-0.00583</td>
    <td>49.174</td>
    <td>17.519</td>
    <td>2.806895</td>
  </tr>
  <tr>
    <td>BERT large SQuAD (ONNX Model Zoo)</td>
    <td>pb</td>
    <td>92.36062</td>
    <td>92.98047</td>
    <td>-0.00667</td>
    <td>45.059</td>
    <td>17.55</td>
    <td>2.567464</td>
  </tr>
  <tr>
    <td>Transformer LT</td>
    <td>pb</td>
    <td>25.815</td>
    <td>25.855</td>
    <td>-0.00155</td>
    <td>28.988</td>
    <td>15.774</td>
    <td>1.837708</td>
  </tr>
  <tr>
    <td>Transformer lt MLPerf</td>
    <td>pb</td>
    <td>27.12967</td>
    <td>27.16596</td>
    <td>-0.00134</td>
    <td>10.269</td>
    <td>5.08</td>
    <td>2.021457</td>
  </tr>
</tbody>
</table>

### Keras Models with keras 2.15.1

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Example</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance 1s4c14ins1bs<br>Throughput(samples/sec)</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio<br>[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio<br>[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Inception ResNet V2</td>
    <td>pb</td>
    <td>0.80252</td>
    <td>0.80398</td>
    <td>-0.00182</td>
    <td>313.966</td>
    <td>178.274</td>
    <td>1.761143</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>pb</td>
    <td>0.7672</td>
    <td>0.76746</td>
    <td>-0.00034</td>
    <td>1076.103</td>
    <td>401.885</td>
    <td>2.677639</td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>pb</td>
    <td>0.71492</td>
    <td>0.71756</td>
    <td>-0.00368</td>
    <td>947.439</td>
    <td>779.506</td>
    <td>1.215435</td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>pb</td>
    <td>0.77524</td>
    <td>0.76448</td>
    <td>0.014075</td>
    <td>1048.361</td>
    <td>384.018</td>
    <td>2.729979</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>0.6909</td>
    <td>0.69028</td>
    <td>0.000898</td>
    <td>411.79</td>
    <td>284.534</td>
    <td>1.447244</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>0.7807</td>
    <td>0.7812</td>
    <td>-0.00064</td>
    <td>680.56</td>
    <td>498.082</td>
    <td>1.366361</td>
  </tr>
  <tr>
    <td>ResNetV2 101</td>
    <td>pb</td>
    <td>0.72644</td>
    <td>0.7187</td>
    <td>0.010769</td>
    <td>492.002</td>
    <td>295.766</td>
    <td>1.663484</td>
  </tr>
  <tr>
    <td>ResNetV2 50</td>
    <td>pb</td>
    <td>0.70374</td>
    <td>0.69642</td>
    <td>0.010511</td>
    <td>779.416</td>
    <td>539.535</td>
    <td>1.444607</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>pb</td>
    <td>0.72692</td>
    <td>0.70886</td>
    <td>0.025478</td>
    <td>1388.622</td>
    <td>203.393</td>
    <td>6.827285</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>pb</td>
    <td>0.72672</td>
    <td>0.71014</td>
    <td>0.023348</td>
    <td>1236.115</td>
    <td>169.736</td>
    <td>7.282574</td>
  </tr>
</tbody></table>

### PyTorch Models with Torch 2.3.0+cpu in PTQ Mode

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Example</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance 1s4c14ins1bs<br>Throughput(samples/sec)</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio<br>[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio<br>[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet18</td>
    <td>static</td>
    <td>0.69594</td>
    <td>0.6976</td>
    <td>-0.00238</td>
    <td>1707.522</td>
    <td>602.469</td>
    <td>2.834207</td>
  </tr>
  <tr>
    <td>EfficientNet-B3</td>
    <td>static</td>
    <td>0.77778</td>
    <td>0.78544</td>
    <td>-0.00975</td>
    <td>513.818</td>
    <td>360.022</td>
    <td>1.427185</td>
  </tr>
  <tr>
    <td>PeleeNet</td>
    <td>static</td>
    <td>0.7183</td>
    <td>0.721</td>
    <td>-0.00374</td>
    <td>837.834</td>
    <td>541.657</td>
    <td>1.546798</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>static</td>
    <td>0.75984</td>
    <td>0.76146</td>
    <td>-0.00213</td>
    <td>1135.222</td>
    <td>311.466</td>
    <td>3.64477</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>static</td>
    <td>0.69456</td>
    <td>0.69522</td>
    <td>-0.00095</td>
    <td>948.026</td>
    <td>322.552</td>
    <td>2.939142</td>
  </tr>
  <tr>
    <td>ResNeSt50</td>
    <td>static</td>
    <td>0.80758</td>
    <td>0.8104</td>
    <td>-0.00348</td>
    <td>406.106</td>
    <td>39.656</td>
    <td>10.24072</td>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>static</td>
    <td>0.78922</td>
    <td>0.79308</td>
    <td>-0.00487</td>
    <td>582.215</td>
    <td>106.731</td>
    <td>5.454976</td>
  </tr>
  <tr>
    <td>YOLO V3</td>
    <td>static</td>
    <td>0.55095</td>
    <td>0.54926</td>
    <td>0.003077</td>
    <td>156.287</td>
    <td>60.296</td>
    <td>2.591996</td>
  </tr>
  <tr>
    <td>Roberta base MRPC</td>
    <td>static</td>
    <td>0.931408</td>
    <td>0.935943</td>
    <td>-0.00485</td>
    <td>396.854</td>
    <td>176.802</td>
    <td>2.244624</td>
  </tr>
  <tr>
    <td>CamemBERT base MRPC</td>
    <td>static</td>
    <td>0.885813</td>
    <td>0.892794</td>
    <td>-0.00782</td>
    <td>405.365</td>
    <td>182.871</td>
    <td>2.216672</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>static</td>
    <td>0.906355</td>
    <td>0.902685</td>
    <td>0.004066</td>
    <td>799.05</td>
    <td>346.498</td>
    <td>2.306074</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>dynamic</td>
    <td>0.900169</td>
    <td>0.902685</td>
    <td>-0.00279</td>
    <td>705.909</td>
    <td>348.157</td>
    <td>2.027559</td>
  </tr>
  <tr>
    <td>ALBERT base MRPC</td>
    <td>static</td>
    <td>0.922807</td>
    <td>0.922807</td>
    <td>0</td>
    <td>350.775</td>
    <td>164.319</td>
    <td>2.13472</td>
  </tr>
  <tr>
    <td>Xlm Roberta MRPC</td>
    <td>static</td>
    <td>0.877966</td>
    <td>0.886207</td>
    <td>-0.0093</td>
    <td>396.063</td>
    <td>175.963</td>
    <td>2.250831</td>
  </tr>
  <tr>
    <td>Xlm Roberta MRPC</td>
    <td>dynamic</td>
    <td>0.885417</td>
    <td>0.882353</td>
    <td>0.003472</td>
    <td>381.19</td>
    <td>175.961</td>
    <td>2.166332</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>static</td>
    <td>0.895944</td>
    <td>0.904202</td>
    <td>-0.00913</td>
    <td>402.419</td>
    <td>177.726</td>
    <td>2.264266</td>
  </tr>
  <tr>
    <td>BERT base COLA</td>
    <td>static</td>
    <td>0.534738</td>
    <td>0.533877</td>
    <td>0.001612</td>
    <td>395.246</td>
    <td>177.024</td>
    <td>2.232726</td>
  </tr>
  <tr>
    <td>BERT base STSB</td>
    <td>static</td>
    <td>0.876114</td>
    <td>0.880462</td>
    <td>-0.00494</td>
    <td>397.62</td>
    <td>177.233</td>
    <td>2.243487</td>
  </tr>
  <tr>
    <td>BERT base SST-2</td>
    <td>static</td>
    <td>0.919725</td>
    <td>0.923165</td>
    <td>-0.00373</td>
    <td>407.661</td>
    <td>182.934</td>
    <td>2.228459</td>
  </tr>
  <tr>
    <td>BERT large COLA</td>
    <td>static</td>
    <td>0.633925</td>
    <td>0.633532</td>
    <td>0.00062</td>
    <td>147.862</td>
    <td>56.008</td>
    <td>2.640016</td>
  </tr>
  <tr>
    <td>BERT base RTE</td>
    <td>static</td>
    <td>0.718412</td>
    <td>0.725632</td>
    <td>-0.00995</td>
    <td>397.827</td>
    <td>177.399</td>
    <td>2.242555</td>
  </tr>
  <tr>
    <td>BERT large MRPC</td>
    <td>static</td>
    <td>0.900662</td>
    <td>0.90378</td>
    <td>-0.00345</td>
    <td>146.838</td>
    <td>52.973</td>
    <td>2.77194</td>
  </tr>
  <tr>
    <td>BERT large QNLI</td>
    <td>static</td>
    <td>0.911221</td>
    <td>0.915431</td>
    <td>-0.0046</td>
    <td>394.508</td>
    <td>176.918</td>
    <td>2.229892</td>
  </tr>
  <tr>
    <td>BERT large RTE</td>
    <td>static</td>
    <td>0.736462</td>
    <td>0.740072</td>
    <td>-0.00488</td>
    <td>148.837</td>
    <td>55.834</td>
    <td>2.665705</td>
  </tr>
  <tr>
    <td>Funnel MRPC</td>
    <td>static</td>
    <td>0.919383</td>
    <td>0.922547</td>
    <td>-0.00343</td>
    <td>294.763</td>
    <td>187.413</td>
    <td>1.572799</td>
  </tr>
  <tr>
    <td>BERT large SQuAD</td>
    <td>static</td>
    <td>92.34173</td>
    <td>93.15842</td>
    <td>-0.00877</td>
    <td>50.205</td>
    <td>18.69</td>
    <td>2.686196</td>
  </tr>
  <tr>
    <td>lvwerra/pegasus-samsum</td>
    <td>static</td>
    <td>42.32</td>
    <td>42.6716</td>
    <td>-0.00824</td>
    <td>102.734</td>
    <td>37.993</td>
    <td>2.704024</td>
  </tr>
</tbody>
</table>

### PyTorch Models with Torch 2.3.0+cpu in QAT Mode

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Example</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance 1s4c14ins1bs<br>Throughput(samples/sec)</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio<br>[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio<br>[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet18</td>
    <td>static</td>
    <td>0.69736</td>
    <td>0.6976</td>
    <td>-0.00034</td>
    <td>1717.593</td>
    <td>602.648</td>
    <td>2.850077</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>static</td>
    <td>0.76034</td>
    <td>0.76146</td>
    <td>-0.00147</td>
    <td>1091.623</td>
    <td>305.83</td>
    <td>3.569378</td>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>static</td>
    <td>0.79308</td>
    <td>0.79308</td>
    <td>0</td>
    <td>584.537</td>
    <td>107.382</td>
    <td>5.443529</td>
  </tr>
</tbody>
</table>

### ONNX Models with ONNX Runtime 1.18.1

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Example</th>
    <th colspan="3">Accuracy</th>
    <th colspan="3">Performance 1s4c14ins1bs<br>Throughput(samples/sec)</th>
  </tr>
  <tr>
    <th>INT8</th>
    <th>FP32</th>
    <th>Accuracy Ratio<br>[(INT8-FP32)/FP32]</th>
    <th>INT8</th>
    <th>FP32</th>
    <th>Performance Ratio<br>[INT8/FP32]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>qlinearops</td>
    <td>0.7218</td>
    <td>0.72294</td>
    <td>-0.00158</td>
    <td>1495.719</td>
    <td>715.937</td>
    <td>2.089177</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>qdq</td>
    <td>0.7213</td>
    <td>0.72294</td>
    <td>-0.00227</td>
    <td>1547.302</td>
    <td>717.027</td>
    <td>2.157941</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 MLPerf</td>
    <td>qlinearops</td>
    <td>0.76146</td>
    <td>0.76462</td>
    <td>-0.00413</td>
    <td>1365.556</td>
    <td>718.546</td>
    <td>1.900443</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 MLPerf</td>
    <td>qdq</td>
    <td>0.76128</td>
    <td>0.76462</td>
    <td>-0.00437</td>
    <td>1445.751</td>
    <td>718.957</td>
    <td>2.010901</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.74768</td>
    <td>0.74988</td>
    <td>-0.00293</td>
    <td>1574.384</td>
    <td>749.356</td>
    <td>2.100983</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.74784</td>
    <td>0.74988</td>
    <td>-0.00272</td>
    <td>1564.149</td>
    <td>755.577</td>
    <td>2.070138</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>qlinearops</td>
    <td>0.66552</td>
    <td>0.66688</td>
    <td>-0.00204</td>
    <td>526.569</td>
    <td>162.642</td>
    <td>3.237595</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>qdq</td>
    <td>0.66616</td>
    <td>0.66688</td>
    <td>-0.00108</td>
    <td>520.089</td>
    <td>172.421</td>
    <td>3.01639</td>
  </tr>
  <tr>
    <td>VGG16 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.72366</td>
    <td>0.72396</td>
    <td>-0.00041</td>
    <td>558.812</td>
    <td>162.87</td>
    <td>3.431031</td>
  </tr>
  <tr>
    <td>VGG16 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.72364</td>
    <td>0.72396</td>
    <td>-0.00044</td>
    <td>556.581</td>
    <td>176.917</td>
    <td>3.146001</td>
  </tr>
  <tr>
    <td>MobileNet V3 MLPerf</td>
    <td>qlinearops</td>
    <td>0.7551</td>
    <td>0.7574</td>
    <td>-0.00304</td>
    <td>5421.716</td>
    <td>2578.076</td>
    <td>2.103009</td>
  </tr>
  <tr>
    <td>MobileNet V3 MLPerf</td>
    <td>qdq</td>
    <td>0.7551</td>
    <td>0.7574</td>
    <td>-0.00304</td>
    <td>5382.87</td>
    <td>2567.479</td>
    <td>2.096559</td>
  </tr>
  <tr>
    <td>ShuffleNet V2 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.66126</td>
    <td>0.66364</td>
    <td>-0.00359</td>
    <td>6426.221</td>
    <td>3725.686</td>
    <td>1.724842</td>
  </tr>
  <tr>
    <td>ShuffleNet V2 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.66216</td>
    <td>0.66364</td>
    <td>-0.00223</td>
    <td>6534.24</td>
    <td>3707.735</td>
    <td>1.762327</td>
  </tr>
  <tr>
    <td>GoogleNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.67694</td>
    <td>0.67786</td>
    <td>-0.00136</td>
    <td>1842.904</td>
    <td>1137.578</td>
    <td>1.620024</td>
  </tr>
  <tr>
    <td>GoogleNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.67714</td>
    <td>0.67786</td>
    <td>-0.00106</td>
    <td>1818.994</td>
    <td>1136.37</td>
    <td>1.600706</td>
  </tr>
  <tr>
    <td>SqueezeNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.56486</td>
    <td>0.56868</td>
    <td>-0.00672</td>
    <td>9521.985</td>
    <td>5530.362</td>
    <td>1.721765</td>
  </tr>
  <tr>
    <td>SqueezeNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.56486</td>
    <td>0.56868</td>
    <td>-0.00672</td>
    <td>9391.072</td>
    <td>5519.793</td>
    <td>1.701345</td>
  </tr>
  <tr>
    <td>CaffeNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.56262</td>
    <td>0.56304</td>
    <td>-0.00075</td>
    <td>2949.36</td>
    <td>893.772</td>
    <td>3.299902</td>
  </tr>
  <tr>
    <td>CaffeNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.56258</td>
    <td>0.56304</td>
    <td>-0.00082</td>
    <td>2847.241</td>
    <td>901.154</td>
    <td>3.15955</td>
  </tr>
  <tr>
    <td>AlexNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.54732</td>
    <td>0.54786</td>
    <td>-0.00099</td>
    <td>2070.166</td>
    <td>816.705</td>
    <td>2.534778</td>
  </tr>
  <tr>
    <td>AlexNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.54712</td>
    <td>0.54786</td>
    <td>-0.00135</td>
    <td>2059.133</td>
    <td>844.973</td>
    <td>2.436922</td>
  </tr>
  <tr>
    <td>ZFNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.55826</td>
    <td>0.5596</td>
    <td>-0.00239</td>
    <td>858.761</td>
    <td>461.254</td>
    <td>1.861796</td>
  </tr>
  <tr>
    <td>ZFNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.55868</td>
    <td>0.5596</td>
    <td>-0.00164</td>
    <td>853.765</td>
    <td>457.911</td>
    <td>1.864478</td>
  </tr>
  <tr>
    <td>Inception V1 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.67228</td>
    <td>0.67242</td>
    <td>-0.00021</td>
    <td>1891.362</td>
    <td>1205.948</td>
    <td>1.568361</td>
  </tr>
  <tr>
    <td>Inception V1 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.67228</td>
    <td>0.67242</td>
    <td>-0.00021</td>
    <td>1879.268</td>
    <td>1202.193</td>
    <td>1.5632</td>
  </tr>
  <tr>
    <td>BEiT (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>85.072</td>
    <td>85.284</td>
    <td>-0.00249</td>
    <td>205.151</td>
    <td>126.587</td>
    <td>1.620632</td>
  </tr>
  <tr>
    <td>EfficientNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.77018</td>
    <td>0.77112</td>
    <td>-0.00122</td>
    <td>2428.315</td>
    <td>1344.025</td>
    <td>1.806748</td>
  </tr>
  <tr>
    <td>EfficientNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.7699</td>
    <td>0.77112</td>
    <td>-0.00158</td>
    <td>2286.726</td>
    <td>1307.18</td>
    <td>1.749358</td>
  </tr>
  <tr>
    <td>DenseNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.60526</td>
    <td>0.60958</td>
    <td>-0.00709</td>
    <td>626.256</td>
    <td>499.762</td>
    <td>1.253108</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.2296</td>
    <td>0.23022</td>
    <td>-0.00269</td>
    <td>1121.431</td>
    <td>841.324</td>
    <td>1.332936</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>0.2296</td>
    <td>0.23022</td>
    <td>-0.00269</td>
    <td>1048.501</td>
    <td>798.218</td>
    <td>1.313552</td>
  </tr>
  <tr>
    <td>DUC (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.81622</td>
    <td>0.81922</td>
    <td>-0.00366</td>
    <td>9.256</td>
    <td>4.988</td>
    <td>1.855654</td>
  </tr>
  <tr>
    <td>Ultra Face (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.83325</td>
    <td>0.83646</td>
    <td>-0.00384</td>
    <td>8993.576</td>
    <td>1988.459</td>
    <td>4.522887</td>
  </tr>
  <tr>
    <td>Emotion FERPlus (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.07941</td>
    <td>0.07997</td>
    <td>-0.007</td>
    <td>6113.744</td>
    <td>3087.501</td>
    <td>1.980159</td>
  </tr>
  <tr>
    <td>ArcFace (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>0.99817</td>
    <td>0.998</td>
    <td>0.00017</td>
    <td>442.845</td>
    <td>230.751</td>
    <td>1.919147</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>qlinearops</td>
    <td>0.85539</td>
    <td>0.86029</td>
    <td>-0.0057</td>
    <td>483.812</td>
    <td>219.454</td>
    <td>2.204617</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>qdq</td>
    <td>0.85539</td>
    <td>0.86029</td>
    <td>-0.0057</td>
    <td>485.079</td>
    <td>218.334</td>
    <td>2.221729</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>integerops</td>
    <td>0.85294</td>
    <td>0.86029</td>
    <td>-0.00854</td>
    <td>684.459</td>
    <td>218.859</td>
    <td>3.127397</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>qdq</td>
    <td>0.84069</td>
    <td>0.84559</td>
    <td>-0.00579</td>
    <td>633.278</td>
    <td>399.309</td>
    <td>1.585935</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>integerops</td>
    <td>0.85539</td>
    <td>0.84559</td>
    <td>0.01159</td>
    <td>1388.435</td>
    <td>401.077</td>
    <td>3.461767</td>
  </tr>
  <tr>
    <td>Mobile bert MRPC</td>
    <td>qdq</td>
    <td>0.85539</td>
    <td>0.86275</td>
    <td>-0.00853</td>
    <td>505.624</td>
    <td>387.431</td>
    <td>1.305069</td>
  </tr>
  <tr>
    <td>Mobile bert MRPC</td>
    <td>integerops</td>
    <td>0.85539</td>
    <td>0.86275</td>
    <td>-0.00853</td>
    <td>565.463</td>
    <td>386.39</td>
    <td>1.463451</td>
  </tr>
  <tr>
    <td>Roberta base MRPC</td>
    <td>integerops</td>
    <td>0.90931</td>
    <td>0.89951</td>
    <td>0.010895</td>
    <td>702.165</td>
    <td>219.495</td>
    <td>3.199002</td>
  </tr>
  <tr>
    <td>BERT SQuAD (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>80.29328</td>
    <td>80.67171</td>
    <td>-0.00469</td>
    <td>242.582</td>
    <td>97.712</td>
    <td>2.482622</td>
  </tr>
  <tr>
    <td>MobileBERT SQuAD MLPerf (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>89.87354</td>
    <td>90.0265</td>
    <td>-0.0017</td>
    <td>151.685</td>
    <td>125.349</td>
    <td>1.210101</td>
  </tr>
  <tr>
    <td>GPT2 lm head WikiText (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>31.98448</td>
    <td>28.99614</td>
    <td>0.10306</td>
    <td>17.964</td>
    <td>10.207</td>
    <td>1.759969</td>
  </tr>
  <tr>
    <td>BERT base uncased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.9021</td>
    <td>0.9042</td>
    <td>-0.00232</td>
    <td>434.653</td>
    <td>210.582</td>
    <td>2.064056</td>
  </tr>
  <tr>
    <td>BERT base uncased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.89577</td>
    <td>0.9042</td>
    <td>-0.00932</td>
    <td>708.657</td>
    <td>210.743</td>
    <td>3.36266</td>
  </tr>
  <tr>
    <td>Roberta base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.91002</td>
    <td>0.91379</td>
    <td>-0.00413</td>
    <td>431.371</td>
    <td>211.03</td>
    <td>2.044122</td>
  </tr>
  <tr>
    <td>Roberta base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.90846</td>
    <td>0.91379</td>
    <td>-0.00583</td>
    <td>711.112</td>
    <td>210.711</td>
    <td>3.374821</td>
  </tr>
  <tr>
    <td>XLM Roberta base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.89369</td>
    <td>0.90102</td>
    <td>-0.00814</td>
    <td>334.877</td>
    <td>211.561</td>
    <td>1.582886</td>
  </tr>
  <tr>
    <td>XLM Roberta base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.89655</td>
    <td>0.90102</td>
    <td>-0.00496</td>
    <td>401.993</td>
    <td>211.429</td>
    <td>1.901314</td>
  </tr>
  <tr>
    <td>Camembert base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.89279</td>
    <td>0.89279</td>
    <td>0</td>
    <td>282.3</td>
    <td>213.326</td>
    <td>1.323327</td>
  </tr>
  <tr>
    <td>Camembert base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.89189</td>
    <td>0.89279</td>
    <td>-0.00101</td>
    <td>707.223</td>
    <td>214.228</td>
    <td>3.301263</td>
  </tr>
  <tr>
    <td>MiniLM L12 H384 uncased MRPC&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.90126</td>
    <td>0.90973</td>
    <td>-0.00931</td>
    <td>1188.05</td>
    <td>578.349</td>
    <td>2.054209</td>
  </tr>
  <tr>
    <td>MiniLM L12 H384 uncased MRPC&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>0.91068</td>
    <td>0.90973</td>
    <td>0.001044</td>
    <td>1285.128</td>
    <td>576.037</td>
    <td>2.230982</td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.90711</td>
    <td>0.91055</td>
    <td>-0.00378</td>
    <td>1259.685</td>
    <td>396.599</td>
    <td>3.176218</td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>0.90252</td>
    <td>0.91055</td>
    <td>-0.00882</td>
    <td>914.626</td>
    <td>395.085</td>
    <td>2.315011</td>
  </tr>
  <tr>
    <td>Albert base v2 SST-2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.92087</td>
    <td>0.92317</td>
    <td>-0.00249</td>
    <td>284.622</td>
    <td>210.521</td>
    <td>1.351989</td>
  </tr>
  <tr>
    <td>Albert base v2 SST-2 (HuggingFace)</td>
    <td>integerops</td>
    <td>0.91743</td>
    <td>0.92317</td>
    <td>-0.00622</td>
    <td>284.686</td>
    <td>209.995</td>
    <td>1.35568</td>
  </tr>
  <tr>
    <td>MiniLM L6 H384 uncased SST-2&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.8945</td>
    <td>0.90138</td>
    <td>-0.00763</td>
    <td>2172.983</td>
    <td>1121.66</td>
    <td>1.937292</td>
  </tr>
  <tr>
    <td>MiniLM L6 H384 uncased SST-2&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>0.89908</td>
    <td>0.90138</td>
    <td>-0.00255</td>
    <td>2326.265</td>
    <td>1114.572</td>
    <td>2.087137</td>
  </tr>
  <tr>
    <td>BERT base cased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.87702</td>
    <td>0.88294</td>
    <td>-0.0067</td>
    <td>494.961</td>
    <td>210.795</td>
    <td>2.348068</td>
  </tr>
  <tr>
    <td>BERT base cased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.88186</td>
    <td>0.88294</td>
    <td>-0.00122</td>
    <td>714.614</td>
    <td>210.991</td>
    <td>3.386941</td>
  </tr>
  <tr>
    <td>Electra small discriminator MRPC&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.89915</td>
    <td>0.89831</td>
    <td>0.000935</td>
    <td>1998.713</td>
    <td>1115.184</td>
    <td>1.792272</td>
  </tr>
  <tr>
    <td>Electra small discriminator MRPC&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>0.89267</td>
    <td>0.89831</td>
    <td>-0.00628</td>
    <td>2202.812</td>
    <td>1121.406</td>
    <td>1.96433</td>
  </tr>
  <tr>
    <td>BERT mini MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.86213</td>
    <td>0.86515</td>
    <td>-0.00349</td>
    <td>5767.229</td>
    <td>3254.792</td>
    <td>1.771919</td>
  </tr>
  <tr>
    <td>BERT mini MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.86159</td>
    <td>0.86515</td>
    <td>-0.00411</td>
    <td>6354.655</td>
    <td>3424.423</td>
    <td>1.855686</td>
  </tr>
  <tr>
    <td>Xlnet base cased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.90052</td>
    <td>0.8986</td>
    <td>0.002137</td>
    <td>121.243</td>
    <td>95.562</td>
    <td>1.268737</td>
  </tr>
  <tr>
    <td>Xlnet base cased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.89583</td>
    <td>0.8986</td>
    <td>-0.00308</td>
    <td>123.055</td>
    <td>95.598</td>
    <td>1.287213</td>
  </tr>
  <tr>
    <td>BART large MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.9236</td>
    <td>0.91197</td>
    <td>0.012753</td>
    <td>126.136</td>
    <td>51.055</td>
    <td>2.470591</td>
  </tr>
  <tr>
    <td>DeBERTa v3 base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>0.92386</td>
    <td>0.92226</td>
    <td>0.001735</td>
    <td>193.159</td>
    <td>153.157</td>
    <td>1.261183</td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>91.1379</td>
    <td>91.97553</td>
    <td>-0.00911</td>
    <td>81.964</td>
    <td>43.36</td>
    <td>1.890314</td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>91.39835</td>
    <td>91.97553</td>
    <td>-0.00628</td>
    <td>101.712</td>
    <td>43.365</td>
    <td>2.345486</td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>88.4205</td>
    <td>89.12625</td>
    <td>-0.00792</td>
    <td>86.328</td>
    <td>43.269</td>
    <td>1.995147</td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>88.70224</td>
    <td>89.12625</td>
    <td>-0.00476</td>
    <td>101.78</td>
    <td>43.244</td>
    <td>2.353621</td>
  </tr>
  <tr>
    <td>DistilBert base uncased SQuAD&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>86.32823</td>
    <td>86.86496</td>
    <td>-0.00618</td>
    <td>120.709</td>
    <td>69.723</td>
    <td>1.731265</td>
  </tr>
  <tr>
    <td>DistilBert base uncased SQuAD&nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>86.04866</td>
    <td>86.86496</td>
    <td>-0.0094</td>
    <td>203.708</td>
    <td>69.682</td>
    <td>2.923395</td>
  </tr>
  <tr>
    <td>BERT large uncased whole word masking&nbsp;&nbsp;&nbsp;SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>92.33945</td>
    <td>93.15547</td>
    <td>-0.00876</td>
    <td>31.813</td>
    <td>12.939</td>
    <td>2.458691</td>
  </tr>
  <tr>
    <td>BERT large uncased whole word masking&nbsp;&nbsp;&nbsp;SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>92.99229</td>
    <td>93.15547</td>
    <td>-0.00175</td>
    <td>35.828</td>
    <td>12.937</td>
    <td>2.769421</td>
  </tr>
  <tr>
    <td>Roberta large SQuAD v2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.03395</td>
    <td>89.01619</td>
    <td>0.0002</td>
    <td>17.61</td>
    <td>13.268</td>
    <td>1.327254</td>
  </tr>
  <tr>
    <td>Roberta large SQuAD v2 (HuggingFace)</td>
    <td>integerops</td>
    <td>89.03833</td>
    <td>89.01619</td>
    <td>0.000249</td>
    <td>35.854</td>
    <td>13.259</td>
    <td>2.704125</td>
  </tr>
  <tr>
    <td>GPT2 WikiText (HuggingFace)</td>
    <td>qlinearops</td>
    <td>30.25292</td>
    <td>28.99614</td>
    <td>0.043343</td>
    <td>13.849</td>
    <td>10.167</td>
    <td>1.362152</td>
  </tr>
  <tr>
    <td>GPT2 WikiText (HuggingFace)</td>
    <td>integerops</td>
    <td>29.68153</td>
    <td>28.99614</td>
    <td>0.023637</td>
    <td>14.636</td>
    <td>10.088</td>
    <td>1.450833</td>
  </tr>
  <tr>
    <td>DistilGPT2 WikiText (HuggingFace)</td>
    <td>qlinearops</td>
    <td>44.93232</td>
    <td>43.43043</td>
    <td>0.034582</td>
    <td>21.798</td>
    <td>17.129</td>
    <td>1.272579</td>
  </tr>
  <tr>
    <td>DistilGPT2 WikiText (HuggingFace)</td>
    <td>integerops</td>
    <td>44.6195</td>
    <td>43.43043</td>
    <td>0.027379</td>
    <td>23.017</td>
    <td>17.086</td>
    <td>1.347126</td>
  </tr>
  <tr>
    <td>LayoutLMv3 FUNSD (HuggingFace)</td>
    <td>integerops</td>
    <td>0.90072</td>
    <td>0.90487</td>
    <td>-0.00459</td>
    <td>39.498</td>
    <td>28</td>
    <td>1.410643</td>
  </tr>
  <tr>
    <td>CodeBert &nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>qlinearops</td>
    <td>0.64971</td>
    <td>0.6541</td>
    <td>-0.00671</td>
    <td>75.685</td>
    <td>45.097</td>
    <td>1.678271</td>
  </tr>
  <tr>
    <td>CodeBert &nbsp;&nbsp;&nbsp;(HuggingFace)</td>
    <td>integerops</td>
    <td>0.64934</td>
    <td>0.6541</td>
    <td>-0.00728</td>
    <td>94.472</td>
    <td>45.103</td>
    <td>2.094584</td>
  </tr>
</tbody></table>

## Validated Pruning Examples

<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Task</br>Dataset</th>
    <th rowspan="2">Dense Accuracy<br>Sparse Accuracy</th>
    <th rowspan="2">Relative Drop</th>
    <th rowspan="2">Sparsity ratio<br>Sparsity Pattern</th>
    <th rowspan="2">Comments<br>Balanced<br>or unbalanced ratio</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>  
  <tr>
    <td>Bert-Mini</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=76.87</br>f1=76.2</td>
    <td>-0.80%</td>
    <td>80%</br>structured 4x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  </tr>  
  <tr>
    <td>Bert-Mini</td>
    <td>question answering</br>SQuAD-v1.1</td>
    <td>f1=76.87</br>f1=76.2</td>
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
    <td>f1=86.90</br>f1=86.15</td>
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
    <td>f1=88.59</br>f1=87.78</td>
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
    <td>f1=91.23</br>f1=90.91</td>
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
  <tr>
    <td>ResNet50</td>
    <td>image recognition</br>ImageNet</td>
    <td>top1 acc = 78.95</br>top1 acc = 80.10</td>
    <td>-1.43%</td>
    <td>75%</br>structured 2x1</td>
    <td>snip momentum</br>unbalanced</td>
  </tr>
  <tr>
  <tr>
    <td>YOLO-v5s6</td>
    <td>object detection</br>COCO</td>
    <td>AP0.50:0.95/AP0.50=0.404/0.6</br>AP0.50:0.95/AP0.50=0.393/0.584</td>
    <td>-2.72%</td>
    <td>80%</br>unstructured</td>
    <td>snip momentum</br>unbalanced</td>
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
</tbody>
</table>

## Validated Knowledge Distillation Examples

| Example Name          | Dataset   | Student<br>(Metrics)                | Teacher<br>(Metrics)               | Student With Distillation<br>(Metrics Improvement) | Student With <br>Distributed Distillation<br>(Metrics Improvement) |
| --------------------- | --------- | ----------------------------------- | ---------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| MobileNet example     | CIFAR-10  | MobileNetV2-0.35<br>(0.7965 ACC)    | WideResNet40-2<br>(0.9522 ACC)     | 0.8178 ACC<br>(0.0213 ACC)                         | 0.8235 ACC<br>(0.027 ACC)                                          |
| CNN example           | CIFAR-100 | CNN-2<br>(0.5494 ACC)               | CNN-10<br>(0.7153 ACC)             | 0.5540 ACC<br>(0.0046 ACC)                         | 0.5523 ACC<br>(0.0029 ACC)                                         |
| VGG example           | CIFAR-100 | VGG-8-BN<br>(0.7022 ACC)            | VGG-13-BN<br>(0.7415 ACC)          | 0.7025 ACC<br>(0.0003 ACC)                         | NA                                                                 |
| ResNet example        | ImageNet  | ResNet18<br>(0.6739 ACC)            | ResNet50<br>(0.7399 ACC)           | 0.6845 ACC<br>(0.0106 ACC)                         | NA                                                                 |
| BlendCnn example      | MRPC      | BlendCnn<br>(0.7034 ACC)            | BERT-Base<br>(0.8382 ACC)          | 0.7034 ACC<br>(0 ACC)                              | NA                                                                 |
| BiLSTM example        | SST-2     | BiLSTM<br>(0.8314 ACC)              | RoBERTa-Base<br>(0.9403 ACC)       | 0.9048 ACC<br>(0.0734 ACC)                         | NA                                                                 |
| DistilBERT example    | SQuAD     | DistilBERT<br>(0.7323/0.8256 EM/F1) | BERT-Base<br>(0.8084/0.8814 EM/F1) | 0.7442/0.8371 EM/F1<br>(0.0119/0.0115 EM/F1)       | NA                                                                 |
| TinyBERT example      | MNLI      | TinyBERT<br>(0.8018/0.8044 m/mm)    | BERT-Base<br>(0.8363/0.8411 m/mm)  | 0.8025/0.8074 m/mm<br>(0.0007/0.0030 m/mm)         | NA                                                                 |
| BERT-3 example        | QQP       | BERT-3<br>(0.8626/0.8213 EM/F1)     | BERT-Base<br>(0.9091/0.8782 EM/F1) | 0.8684/0.8259 EM/F1<br>(0.0058/0.0046 EM/F1)       | NA                                                                 |
| DistilRoBERTa example | COLA      | DistilRoBERTa<br>(0.6057 ACC)       | RoBERTa-Large<br>(0.6455 ACC)      | 0.6187 ACC<br>(0.0130 ACC)                         | NA                                                                 |

## Validated ONNX QDQ INT8 Models on Multiple Hardware through ONNX Runtime

<table class="docutils">
<thead>
  <tr>
    <th class="tg-y3we">Model (ONNX QDQ)</th>
    <th class="tg-pm1l">AWS c6i.2xlarge (Intel)<br>CPU Execution Provider</th>
    <th class="tg-pm1l">AWS c6a.2xlarge (AMD)<br>CPU Execution Provider</th>
    <th class="tg-pm1l">AWS c6g.2xlarge (ARM)<br>CPU Execution Provider</th>
    <th class="tg-8d8j">NVidia A100<br>CUDA Execution<br>Provider</th>
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

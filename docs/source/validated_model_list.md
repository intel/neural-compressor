# Validated Models

Intel® Neural Compressor validated examples with multiple compression techniques. The typical examples link can be found in [example tables](https://github.com/intel/neural-compressor/blob/master/examples/README.md), and the performance/accuracy results is available here.

1. [Validated Quantization Examples](#Validated-Quantization-Examples)

   1.1. [TensorFlow Models with TensorFlow 2.16.1](#tensorflow-models-with-tensorflow-2161)

   1.2. [Keras Models with keras 2.15.1](#keras-models-with-keras-2151)

   1.3. [PyTorch Models with Torch 2.3.0+cpu in PTQ Mode](#pytorch-models-with-torch-230cpu-in-ptq-mode)

   1.4. [PyTorch Models with Torch 2.3.0+cpu in QAT Mode](#pytorch-models-with-torch-230cpu-in-qat-mode)

   1.5. [PyTorch Models with Torch 2.3.0+cpu in IPEX Mode](#pytorch-models-with-torch-230cpu-in-ipex-mode)

   1.6. [ONNX Models with ONNX Runtime 1.18.1](#onnx-models-with-onnx-runtime-1181)

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
    <td>74.11%</td>
    <td>74.27%</td>
    <td>-0.22%</td>
    <td>1732.92</td>
    <td>578.88</td>
    <td>2.99x</td>
  </tr>
  <tr>
    <td>ResNet50 v1.5</td>
    <td>pb</td>
    <td>76.25%</td>
    <td>76.46%</td>
    <td>-0.28%</td>
    <td>1535.20</td>
    <td>530.00</td>
    <td>2.90x</td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>pb</td>
    <td>77.52%</td>
    <td>76.45%</td>
    <td>1.41%</td>
    <td>1048.36</td>
    <td>384.02</td>
    <td>2.73x</td>
  </tr>
  <tr>
    <td>Inception V1</td>
    <td>pb</td>
    <td>70.45%</td>
    <td>69.74%</td>
    <td>+1.03%</td>
    <td>2079.24</td>
    <td>927.82</td>
    <td>2.24x</td>
  </tr>
  <tr>
    <td>Inception V2</td>
    <td>pb</td>
    <td>74.33%</td>
    <td>73.97%</td>
    <td>+0.49%</td>
    <td>1644.36</td>
    <td>840.53</td>
    <td>1.96x</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>pb</td>
    <td>76.72%</td>
    <td>76.75%</td>
    <td>-0.03%</td>
    <td>1076.10</td>
    <td>401.89</td>
    <td>2.68x</td>
  </tr>
  <tr>
    <td>Inception V4</td>
    <td>pb</td>
    <td>80.13%</td>
    <td>80.27%</td>
    <td>-0.18%</td>
    <td>704.96</td>
    <td>199.28</td>
    <td>3.54x</td>
  </tr>
  <tr>
    <td>Inception ResNet V2</td>
    <td>pb</td>
    <td>80.25%</td>
    <td>80.40%</td>
    <td>-0.18%</td>
    <td>313.97</td>
    <td>178.27</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>DenseNet-161</td>
    <td>pb</td>
    <td>76.29%</td>
    <td>76.29%</td>
    <td>+0.00%</td>
    <td>279.20</td>
    <td>214.03</td>
    <td>1.30x</td>
  </tr>
  <tr>
    <td>MobileNet V1</td>
    <td>pb</td>
    <td>71.79%</td>
    <td>70.96%</td>
    <td>+1.18%</td>
    <td>4199.13</td>
    <td>1506.68</td>
    <td>2.79x</td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>pb</td>
    <td>72.48%</td>
    <td>71.76%</td>
    <td>+1.01%</td>
    <td>2170.39</td>
    <td>1445.05</td>
    <td>1.50x</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>pb</td>
    <td>72.69%</td>
    <td>70.89%</td>
    <td>+2.55%</td>
    <td>1388.62</td>
    <td>203.39</td>
    <td>6.83x</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>pb</td>
    <td>72.67%</td>
    <td>71.01%</td>
    <td>+2.33%</td>
    <td>1236.12</td>
    <td>169.74</td>
    <td>7.28x</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>69.09%</td>
    <td>69.03%</td>
    <td>+0.09%</td>
    <td>411.79</td>
    <td>284.53</td>
    <td>1.45x</td>
  </tr>
  <tr>
    <td>ResNetV2 50</td>
    <td>pb</td>
    <td>70.37%</td>
    <td>69.64%</td>
    <td>+1.05%</td>
    <td>779.42</td>
    <td>539.54</td>
    <td>1.44x</td>
  </tr>
  <tr>
    <td>ResNetV2 101</td>
    <td>pb</td>
    <td>72.64%</td>
    <td>71.87%</td>
    <td>+1.08%</td>
    <td>492.00</td>
    <td>295.77</td>
    <td>1.66x</td>
  </tr>
  <tr>
    <td>ResNetV2 152</td>
    <td>pb</td>
    <td>73.12%</td>
    <td>72.37%</td>
    <td>+1.04%</td>
    <td>348.39</td>
    <td>205.72</td>
    <td>1.69x</td>
  </tr>
  <tr>
    <td>ViT</td>
    <td>pb</td>
    <td>81.39%</td>
    <td>81.92%</td>
    <td>-0.64%</td>
    <td>230.53</td>
    <td>132.66</td>
    <td>1.74x</td>
  </tr>
  <tr>
    <td>SSD ResNet50 V1</td>
    <td>pb</td>
    <td>37.91%</td>
    <td>38.00%</td>
    <td>-0.24%</td>
    <td>135.71</td>
    <td>28.75</td>
    <td>4.72x</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1</td>
    <td>pb</td>
    <td>23.00%</td>
    <td>23.13%</td>
    <td>-0.57%</td>
    <td>1237.70</td>
    <td>719.30</td>
    <td>1.72x</td>
  </tr>
  <tr>
    <td>SSD ResNet50 v1</td>
    <td>ckpt</td>
    <td>37.88%</td>
    <td>38.00%</td>
    <td>-0.31%</td>
    <td>130.54</td>
    <td>22.05</td>
    <td>5.92x</td>
  </tr>
  <tr>
    <td>SSD MobileNet v1</td>
    <td>ckpt</td>
    <td>22.96%</td>
    <td>23.13%</td>
    <td>-0.71%</td>
    <td>1234.56</td>
    <td>529.34</td>
    <td>2.33x</td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet101</td>
    <td>pb</td>
    <td>30.32%</td>
    <td>30.39%</td>
    <td>-0.22%</td>
    <td>144.21</td>
    <td>22.64</td>
    <td>6.37x</td>
  </tr>
  <tr>
    <td>Faster R-CNN ResNet50</td>
    <td>pb</td>
    <td>26.61%</td>
    <td>26.59%</td>
    <td>+0.09%</td>
    <td>164.55</td>
    <td>28.38</td>
    <td>5.80x</td>
  </tr>
  <tr>
    <td>YOLOv3</td>
    <td>pb</td>
    <td>83.28%</td>
    <td>82.35%</td>
    <td>+1.12%</td>
    <td>247.56</td>
    <td>81.45</td>
    <td>3.04x</td>
  </tr>
  <tr>
    <td>BERT large SQuAD</td>
    <td>pb</td>
    <td>92.44%</td>
    <td>92.99%</td>
    <td>-0.58%</td>
    <td>49.17</td>
    <td>17.52</td>
    <td>2.81x</td>
  </tr>
  <tr>
    <td>BERT large SQuAD (ONNX Model Zoo)</td>
    <td>pb</td>
    <td>92.36%</td>
    <td>92.98%</td>
    <td>-0.67%</td>
    <td>45.06</td>
    <td>17.55</td>
    <td>2.57x</td>
  </tr>
  <tr>
    <td>Transformer LT</td>
    <td>pb</td>
    <td>25.82%</td>
    <td>25.86%</td>
    <td>-0.15%</td>
    <td>28.99</td>
    <td>15.77</td>
    <td>1.84x</td>
  </tr>
  <tr>
    <td>Transformer lt MLPerf</td>
    <td>pb</td>
    <td>27.13%</td>
    <td>27.17%</td>
    <td>-0.13%</td>
    <td>10.27</td>
    <td>5.08</td>
    <td>2.02x</td>
  </tr>
    <tr>
    <td>Mask R-CNN Inception V2</td>
    <td>pb</td>
    <td>28.46%</td>
    <td>28.73%</td>
    <td>-0.91%</td>
    <td>195.68</td>
    <td>50.72</td>
    <td>3.86x</td>
  </tr>
  <tr>
    <td>Mask R-CNN Inception V2</td>
    <td>ckpt</td>
    <td>28.46%</td>
    <td>28.73%</td>
    <td>-0.91%</td>
    <td>206.14</td>
    <td>47.04</td>
    <td>4.38x</td>
  </tr>
</tbody></table>

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
    <td>80.25%</td>
    <td>80.40%</td>
    <td>-0.18%</td>
    <td>313.97</td>
    <td>178.27</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>pb</td>
    <td>76.72%</td>
    <td>76.75%</td>
    <td>-0.03%</td>
    <td>1076.10</td>
    <td>401.89</td>
    <td>2.68x</td>
  </tr>
  <tr>
    <td>MobileNet V2</td>
    <td>pb</td>
    <td>71.49%</td>
    <td>71.76%</td>
    <td>-0.37%</td>
    <td>947.44</td>
    <td>779.51</td>
    <td>1.22x</td>
  </tr>
  <tr>
    <td>ResNet101</td>
    <td>pb</td>
    <td>77.52%</td>
    <td>76.45%</td>
    <td>+1.41%</td>
    <td>1048.36</td>
    <td>384.02</td>
    <td>2.73x</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>69.09%</td>
    <td>69.03%</td>
    <td>+0.09%</td>
    <td>411.79</td>
    <td>284.53</td>
    <td>1.45x</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>pb</td>
    <td>78.07%</td>
    <td>78.12%</td>
    <td>-0.06%</td>
    <td>680.56</td>
    <td>498.08</td>
    <td>1.37x</td>
  </tr>
  <tr>
    <td>ResNetV2 101</td>
    <td>pb</td>
    <td>72.64%</td>
    <td>71.87%</td>
    <td>+1.08%</td>
    <td>492.00</td>
    <td>295.77</td>
    <td>1.66x</td>
  </tr>
  <tr>
    <td>ResNetV2 50</td>
    <td>pb</td>
    <td>70.37%</td>
    <td>69.64%</td>
    <td>+1.05%</td>
    <td>779.42</td>
    <td>539.54</td>
    <td>1.44x</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>pb</td>
    <td>72.69%</td>
    <td>70.89%</td>
    <td>+2.55%</td>
    <td>1388.62</td>
    <td>203.39</td>
    <td>6.83x</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>pb</td>
    <td>72.67%</td>
    <td>71.01%</td>
    <td>+2.33%</td>
    <td>1236.12</td>
    <td>169.74</td>
    <td>7.28x</td>
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
    <td>69.59%</td>
    <td>69.76%</td>
    <td>-0.24%</td>
    <td>1707.52</td>
    <td>602.47</td>
    <td>2.83x</td>
  </tr>
  <tr>
    <td>EfficientNet-B3</td>
    <td>static</td>
    <td>77.78%</td>
    <td>78.54%</td>
    <td>-0.98%</td>
    <td>513.82</td>
    <td>360.02</td>
    <td>1.43x</td>
  </tr>
  <tr>
    <td>PeleeNet</td>
    <td>static</td>
    <td>71.83%</td>
    <td>72.10%</td>
    <td>-0.37%</td>
    <td>837.83</td>
    <td>541.66</td>
    <td>1.55x</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>static</td>
    <td>75.98%</td>
    <td>76.15%</td>
    <td>-0.21%</td>
    <td>1135.22</td>
    <td>311.47</td>
    <td>3.64x</td>
  </tr>
  <tr>
    <td>Inception V3</td>
    <td>static</td>
    <td>69.46%</td>
    <td>69.52%</td>
    <td>-0.09%</td>
    <td>948.03</td>
    <td>322.55</td>
    <td>2.94x</td>
  </tr>
  <tr>
    <td>ResNeSt50</td>
    <td>static</td>
    <td>80.76%</td>
    <td>81.04%</td>
    <td>-0.35%</td>
    <td>406.11</td>
    <td>39.66</td>
    <td>10.24x</td>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>static</td>
    <td>78.92%</td>
    <td>79.31%</td>
    <td>-0.49%</td>
    <td>582.22</td>
    <td>106.73</td>
    <td>5.45x</td>
  </tr>
  <tr>
    <td>YOLO V3</td>
    <td>static</td>
    <td>55.10%</td>
    <td>54.93%</td>
    <td>+0.31%</td>
    <td>156.29</td>
    <td>60.30</td>
    <td>2.59x</td>
  </tr>
  <tr>
    <td>Roberta base MRPC</td>
    <td>static</td>
    <td>93.14%</td>
    <td>93.59%</td>
    <td>-0.48%</td>
    <td>396.85</td>
    <td>176.80</td>
    <td>2.24x</td>
  </tr>
  <tr>
    <td>CamemBERT base MRPC</td>
    <td>static</td>
    <td>88.58%</td>
    <td>89.28%</td>
    <td>-0.78%</td>
    <td>405.37</td>
    <td>182.87</td>
    <td>2.22x</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>static</td>
    <td>90.64%</td>
    <td>90.27%</td>
    <td>+0.41%</td>
    <td>799.05</td>
    <td>346.50</td>
    <td>2.31x</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>dynamic</td>
    <td>90.02%</td>
    <td>90.27%</td>
    <td>-0.28%</td>
    <td>705.91</td>
    <td>348.16</td>
    <td>2.03x</td>
  </tr>
  <tr>
    <td>ALBERT base MRPC</td>
    <td>static</td>
    <td>92.28%</td>
    <td>92.28%</td>
    <td>0.00%</td>
    <td>350.78</td>
    <td>164.32</td>
    <td>2.13x</td>
  </tr>
  <tr>
    <td>Xlm Roberta MRPC</td>
    <td>static</td>
    <td>87.80%</td>
    <td>88.62%</td>
    <td>-0.93%</td>
    <td>396.06</td>
    <td>175.96</td>
    <td>2.25x</td>
  </tr>
  <tr>
    <td>Xlm Roberta MRPC</td>
    <td>dynamic</td>
    <td>88.54%</td>
    <td>88.24%</td>
    <td>+0.35%</td>
    <td>381.19</td>
    <td>175.96</td>
    <td>2.17x</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>static</td>
    <td>89.59%</td>
    <td>90.42%</td>
    <td>-0.91%</td>
    <td>402.42</td>
    <td>177.73</td>
    <td>2.26x</td>
  </tr>
  <tr>
    <td>BERT base COLA</td>
    <td>static</td>
    <td>53.47%</td>
    <td>53.39%</td>
    <td>+0.16%</td>
    <td>395.25</td>
    <td>177.02</td>
    <td>2.23x</td>
  </tr>
  <tr>
    <td>BERT base STSB</td>
    <td>static</td>
    <td>87.61%</td>
    <td>88.05%</td>
    <td>-0.49%</td>
    <td>397.62</td>
    <td>177.23</td>
    <td>2.24x</td>
  </tr>
  <tr>
    <td>BERT base SST-2</td>
    <td>static</td>
    <td>91.97%</td>
    <td>92.32%</td>
    <td>-0.37%</td>
    <td>407.66</td>
    <td>182.93</td>
    <td>2.23x</td>
  </tr>
  <tr>
    <td>BERT large COLA</td>
    <td>static</td>
    <td>63.39%</td>
    <td>63.35%</td>
    <td>+0.06%</td>
    <td>147.86</td>
    <td>56.01</td>
    <td>2.64x</td>
  </tr>
  <tr>
    <td>BERT base RTE</td>
    <td>static</td>
    <td>71.84%</td>
    <td>72.56%</td>
    <td>-1.00%</td>
    <td>397.83</td>
    <td>177.40</td>
    <td>2.24x</td>
  </tr>
  <tr>
    <td>BERT large MRPC</td>
    <td>static</td>
    <td>90.07%</td>
    <td>90.38%</td>
    <td>-0.34%</td>
    <td>146.84</td>
    <td>52.97</td>
    <td>2.77x</td>
  </tr>
  <tr>
    <td>BERT large QNLI</td>
    <td>static</td>
    <td>91.12%</td>
    <td>91.54%</td>
    <td>-0.46%</td>
    <td>394.51</td>
    <td>176.92</td>
    <td>2.23x</td>
  </tr>
  <tr>
    <td>BERT large RTE</td>
    <td>static</td>
    <td>73.65%</td>
    <td>74.01%</td>
    <td>-0.49%</td>
    <td>148.84</td>
    <td>55.83</td>
    <td>2.67x</td>
  </tr>
  <tr>
    <td>Funnel MRPC</td>
    <td></td>
    <td>91.94%</td>
    <td>92.25%</td>
    <td>-0.34%</td>
    <td>294.76</td>
    <td>187.41</td>
    <td>1.57x</td>
  </tr>
  <tr>
    <td>BERT large SQuAD</td>
    <td>static</td>
    <td>92.34%</td>
    <td>93.16%</td>
    <td>-0.88%</td>
    <td>50.21</td>
    <td>18.69</td>
    <td>2.69x</td>
  </tr>
  <tr>
    <td>lvwerra/pegasus-samsum</td>
    <td>static</td>
    <td>42.32%</td>
    <td>42.67%</td>
    <td>-0.82%</td>
    <td>102.73</td>
    <td>37.99</td>
    <td>2.70x</td>
  </tr>
  <tr>
    <td>ResNet18 PT2E</td>
    <td>static</td>
    <td>69.49%</td>
    <td>69.76%</td>
    <td>-0.39%</td>
    <td>1873.51</td>
    <td>1106.97</td>
    <td>1.69x</td>
  </tr>
  <tr>
    <td>OPT-125M PT2E</td>
    <td>static</td>
    <td>37.07%</td>
    <td>37.90%</td>
    <td>-2.20%</td>
    <td>42.09</td>
    <td>29.68</td>
    <td>1.42x</td>
  </tr>
</tbody></table>

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
    <td>69.74%</td>
    <td>69.76%</td>
    <td>-0.03%</td>
    <td>1717.59</td>
    <td>602.65</td>
    <td>2.85x</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>static</td>
    <td>76.03%</td>
    <td>76.15%</td>
    <td>-0.15%</td>
    <td>1091.62</td>
    <td>305.83</td>
    <td>3.57x</td>
  </tr>
  <tr>
    <td>ResNeXt101_32x8d</td>
    <td>static</td>
    <td>79.31%</td>
    <td>79.31%</td>
    <td>0.00%</td>
    <td>584.54</td>
    <td>107.38</td>
    <td>5.44x</td>
  </tr>
</tbody>
</table>

### PyTorch Models with Torch 2.3.0+cpu in IPEX Mode

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
    <td>bert-large-uncased-whole-word-masking-finetuned-squad</td>
    <td>static</td>
    <td>93.01%</td>
    <td>93.16%</td>
    <td>-0.16%</td>
    <td>150.05</td>
    <td>22.42</td>
    <td>6.69x</td>
  </tr>
  <tr>
    <td>distilbert-base-uncased-distilled-squad</td>
    <td>static</td>
    <td>86.10%</td>
    <td>86.84%</td>
    <td>-0.85%</td>
    <td>1034.60</td>
    <td>151.13</td>
    <td>6.85x</td>
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
    <td>72.18%</td>
    <td>72.29%</td>
    <td>-0.16%</td>
    <td>1495.72</td>
    <td>715.94</td>
    <td>2.09x</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5</td>
    <td>qdq</td>
    <td>72.13%</td>
    <td>72.29%</td>
    <td>-0.23%</td>
    <td>1547.30</td>
    <td>717.03</td>
    <td>2.16x</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 MLPerf</td>
    <td>qlinearops</td>
    <td>76.15%</td>
    <td>76.46%</td>
    <td>-0.41%</td>
    <td>1365.56</td>
    <td>718.55</td>
    <td>1.90x</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 MLPerf</td>
    <td>qdq</td>
    <td>76.13%</td>
    <td>76.46%</td>
    <td>-0.44%</td>
    <td>1445.75</td>
    <td>718.96</td>
    <td>2.01x</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>74.77%</td>
    <td>74.99%</td>
    <td>-0.29%</td>
    <td>1574.38</td>
    <td>749.36</td>
    <td>2.10x</td>
  </tr>
  <tr>
    <td>ResNet50 V1.5 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>74.78%</td>
    <td>74.99%</td>
    <td>-0.27%</td>
    <td>1564.15</td>
    <td>755.58</td>
    <td>2.07x</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>qlinearops</td>
    <td>66.55%</td>
    <td>66.69%</td>
    <td>-0.20%</td>
    <td>526.57</td>
    <td>162.64</td>
    <td>3.24x</td>
  </tr>
  <tr>
    <td>VGG16</td>
    <td>qdq</td>
    <td>66.62%</td>
    <td>66.69%</td>
    <td>-0.11%</td>
    <td>520.09</td>
    <td>172.42</td>
    <td>3.02x</td>
  </tr>
  <tr>
    <td>VGG16 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>72.37%</td>
    <td>72.40%</td>
    <td>-0.04%</td>
    <td>558.81</td>
    <td>162.87</td>
    <td>3.43x</td>
  </tr>
  <tr>
    <td>VGG16 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>72.36%</td>
    <td>72.40%</td>
    <td>-0.04%</td>
    <td>556.58</td>
    <td>176.92</td>
    <td>3.15x</td>
  </tr>
  <tr>
    <td>MobileNet V3 MLPerf</td>
    <td>qlinearops</td>
    <td>75.51%</td>
    <td>75.74%</td>
    <td>-0.30%</td>
    <td>5421.72</td>
    <td>2578.08</td>
    <td>2.10x</td>
  </tr>
  <tr>
    <td>MobileNet V3 MLPerf</td>
    <td>qdq</td>
    <td>75.51%</td>
    <td>75.74%</td>
    <td>-0.30%</td>
    <td>5382.87</td>
    <td>2567.48</td>
    <td>2.10x</td>
  </tr>
  <tr>
    <td>ShuffleNet V2 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>66.13%</td>
    <td>66.36%</td>
    <td>-0.36%</td>
    <td>6426.22</td>
    <td>3725.69</td>
    <td>1.72x</td>
  </tr>
  <tr>
    <td>ShuffleNet V2 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>66.22%</td>
    <td>66.36%</td>
    <td>-0.22%</td>
    <td>6534.24</td>
    <td>3707.74</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>GoogleNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>67.69%</td>
    <td>67.79%</td>
    <td>-0.14%</td>
    <td>1842.90</td>
    <td>1137.58</td>
    <td>1.62x</td>
  </tr>
  <tr>
    <td>GoogleNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>67.71%</td>
    <td>67.79%</td>
    <td>-0.11%</td>
    <td>1818.99</td>
    <td>1136.37</td>
    <td>1.60x</td>
  </tr>
  <tr>
    <td>SqueezeNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>56.49%</td>
    <td>56.87%</td>
    <td>-0.67%</td>
    <td>9521.99</td>
    <td>5530.36</td>
    <td>1.72x</td>
  </tr>
  <tr>
    <td>SqueezeNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>56.49%</td>
    <td>56.87%</td>
    <td>-0.67%</td>
    <td>9391.07</td>
    <td>5519.79</td>
    <td>1.70x</td>
  </tr>
  <tr>
    <td>CaffeNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>56.26%</td>
    <td>56.30%</td>
    <td>-0.07%</td>
    <td>2949.36</td>
    <td>893.77</td>
    <td>3.30x</td>
  </tr>
  <tr>
    <td>CaffeNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>56.26%</td>
    <td>56.30%</td>
    <td>-0.08%</td>
    <td>2847.24</td>
    <td>901.15</td>
    <td>3.16x</td>
  </tr>
  <tr>
    <td>AlexNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>54.73%</td>
    <td>54.79%</td>
    <td>-0.10%</td>
    <td>2070.17</td>
    <td>816.71</td>
    <td>2.53x</td>
  </tr>
  <tr>
    <td>AlexNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>54.71%</td>
    <td>54.79%</td>
    <td>-0.14%</td>
    <td>2059.13</td>
    <td>844.97</td>
    <td>2.44x</td>
  </tr>
  <tr>
    <td>ZFNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>55.83%</td>
    <td>55.96%</td>
    <td>-0.24%</td>
    <td>858.76</td>
    <td>461.25</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>ZFNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>55.87%</td>
    <td>55.96%</td>
    <td>-0.16%</td>
    <td>853.77</td>
    <td>457.91</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>Inception V1 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>67.23%</td>
    <td>67.24%</td>
    <td>-0.02%</td>
    <td>1891.36</td>
    <td>1205.95</td>
    <td>1.57x</td>
  </tr>
  <tr>
    <td>Inception V1 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>67.23%</td>
    <td>67.24%</td>
    <td>-0.02%</td>
    <td>1879.27</td>
    <td>1202.19</td>
    <td>1.56x</td>
  </tr>
  <tr>
    <td>BEiT (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>85.07%</td>
    <td>85.28%</td>
    <td>-0.25%</td>
    <td>205.15</td>
    <td>126.59</td>
    <td>1.62x</td>
  </tr>
  <tr>
    <td>EfficientNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>77.02%</td>
    <td>77.11%</td>
    <td>-0.12%</td>
    <td>2428.32</td>
    <td>1344.03</td>
    <td>1.81x</td>
  </tr>
  <tr>
    <td>EfficientNet (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>76.99%</td>
    <td>77.11%</td>
    <td>-0.16%</td>
    <td>2286.73</td>
    <td>1307.18</td>
    <td>1.75x</td>
  </tr>
  <tr>
    <td>DenseNet (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>60.53%</td>
    <td>60.96%</td>
    <td>-0.71%</td>
    <td>626.26</td>
    <td>499.76</td>
    <td>1.25x</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>22.96%</td>
    <td>23.02%</td>
    <td>-0.27%</td>
    <td>1121.43</td>
    <td>841.32</td>
    <td>1.33x</td>
  </tr>
  <tr>
    <td>SSD MobileNet V1 (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>22.96%</td>
    <td>23.02%</td>
    <td>-0.27%</td>
    <td>1048.50</td>
    <td>798.22</td>
    <td>1.31x</td>
  </tr>
  <tr>
    <td>DUC (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>81.62%</td>
    <td>81.92%</td>
    <td>-0.37%</td>
    <td>9.26</td>
    <td>4.99</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>Ultra Face (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>83.33%</td>
    <td>83.65%</td>
    <td>-0.38%</td>
    <td>8993.58</td>
    <td>1988.46</td>
    <td>4.52x</td>
  </tr>
  <tr>
    <td>Emotion FERPlus (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>7.94%</td>
    <td>8.00%</td>
    <td>-0.70%</td>
    <td>6113.74</td>
    <td>3087.50</td>
    <td>1.98x</td>
  </tr>
  <tr>
    <td>ArcFace (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>99.82%</td>
    <td>99.80%</td>
    <td>+0.02%</td>
    <td>442.85</td>
    <td>230.75</td>
    <td>1.92x</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>qlinearops</td>
    <td>85.54%</td>
    <td>86.03%</td>
    <td>-0.57%</td>
    <td>483.81</td>
    <td>219.45</td>
    <td>2.20x</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>qdq</td>
    <td>85.54%</td>
    <td>86.03%</td>
    <td>-0.57%</td>
    <td>485.08</td>
    <td>218.33</td>
    <td>2.22x</td>
  </tr>
  <tr>
    <td>BERT base MRPC</td>
    <td>integerops</td>
    <td>85.29%</td>
    <td>86.03%</td>
    <td>-0.85%</td>
    <td>684.46</td>
    <td>218.86</td>
    <td>3.13x</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>qdq</td>
    <td>84.07%</td>
    <td>84.56%</td>
    <td>-0.58%</td>
    <td>633.28</td>
    <td>399.31</td>
    <td>1.59x</td>
  </tr>
  <tr>
    <td>DistilBERT base MRPC</td>
    <td>integerops</td>
    <td>85.54%</td>
    <td>84.56%</td>
    <td>+1.16%</td>
    <td>1388.44</td>
    <td>401.08</td>
    <td>3.46x</td>
  </tr>
  <tr>
    <td>Mobile bert MRPC</td>
    <td>qdq</td>
    <td>85.54%</td>
    <td>86.28%</td>
    <td>-0.85%</td>
    <td>505.62</td>
    <td>387.43</td>
    <td>1.31x</td>
  </tr>
  <tr>
    <td>Mobile bert MRPC</td>
    <td>integerops</td>
    <td>85.54%</td>
    <td>86.28%</td>
    <td>-0.85%</td>
    <td>565.46</td>
    <td>386.39</td>
    <td>1.46x</td>
  </tr>
  <tr>
    <td>Roberta base MRPC</td>
    <td>integerops</td>
    <td>90.93%</td>
    <td>89.95%</td>
    <td>+1.09%</td>
    <td>702.17</td>
    <td>219.50</td>
    <td>3.20x</td>
  </tr>
  <tr>
    <td>BERT SQuAD (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>80.29%</td>
    <td>80.67%</td>
    <td>-0.47%</td>
    <td>242.58</td>
    <td>97.71</td>
    <td>2.48x</td>
  </tr>
  <tr>
    <td>MobileBERT SQuAD MLPerf (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>89.87%</td>
    <td>90.03%</td>
    <td>-0.17%</td>
    <td>151.69</td>
    <td>125.35</td>
    <td>1.21x</td>
  </tr>
  <tr>
    <td>GPT2 lm head WikiText (ONNX Model Zoo)</td>
    <td>integerops</td>
    <td>31.98%</td>
    <td>29.00%</td>
    <td>+10.31%</td>
    <td>17.96</td>
    <td>10.21</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>BERT base uncased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>90.21%</td>
    <td>90.42%</td>
    <td>-0.23%</td>
    <td>434.65</td>
    <td>210.58</td>
    <td>2.06x</td>
  </tr>
  <tr>
    <td>BERT base uncased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>89.58%</td>
    <td>90.42%</td>
    <td>-0.93%</td>
    <td>708.66</td>
    <td>210.74</td>
    <td>3.36x</td>
  </tr>
  <tr>
    <td>Roberta base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>91.00%</td>
    <td>91.38%</td>
    <td>-0.41%</td>
    <td>431.37</td>
    <td>211.03</td>
    <td>2.04x</td>
  </tr>
  <tr>
    <td>Roberta base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>90.85%</td>
    <td>91.38%</td>
    <td>-0.58%</td>
    <td>711.11</td>
    <td>210.71</td>
    <td>3.37x</td>
  </tr>
  <tr>
    <td>XLM Roberta base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.37%</td>
    <td>90.10%</td>
    <td>-0.81%</td>
    <td>334.88</td>
    <td>211.56</td>
    <td>1.58x</td>
  </tr>
  <tr>
    <td>XLM Roberta base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>89.66%</td>
    <td>90.10%</td>
    <td>-0.50%</td>
    <td>401.99</td>
    <td>211.43</td>
    <td>1.90x</td>
  </tr>
  <tr>
    <td>Camembert base MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.28%</td>
    <td>89.28%</td>
    <td>0.00%</td>
    <td>282.30</td>
    <td>213.33</td>
    <td>1.32x</td>
  </tr>
  <tr>
    <td>Camembert base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>89.19%</td>
    <td>89.28%</td>
    <td>-0.10%</td>
    <td>707.22</td>
    <td>214.23</td>
    <td>3.30x</td>
  </tr>
  <tr>
    <td>MiniLM L12 H384 uncased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>90.13%</td>
    <td>90.97%</td>
    <td>-0.93%</td>
    <td>1188.05</td>
    <td>578.35</td>
    <td>2.05x</td>
  </tr>
  <tr>
    <td>MiniLM L12 H384 uncased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>91.07%</td>
    <td>90.97%</td>
    <td>+0.10%</td>
    <td>1285.13</td>
    <td>576.04</td>
    <td>2.23x</td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>90.71%</td>
    <td>91.06%</td>
    <td>-0.38%</td>
    <td>1259.69</td>
    <td>396.60</td>
    <td>3.18x</td>
  </tr>
  <tr>
    <td>DistilBERT base uncased SST-2 (HuggingFace)</td>
    <td>integerops</td>
    <td>90.25%</td>
    <td>91.06%</td>
    <td>-0.88%</td>
    <td>914.63</td>
    <td>395.09</td>
    <td>2.32x</td>
  </tr>
  <tr>
    <td>Albert base v2 SST-2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>92.09%</td>
    <td>92.32%</td>
    <td>-0.25%</td>
    <td>284.62</td>
    <td>210.52</td>
    <td>1.35x</td>
  </tr>
  <tr>
    <td>Albert base v2 SST-2 (HuggingFace)</td>
    <td>integerops</td>
    <td>91.74%</td>
    <td>92.32%</td>
    <td>-0.62%</td>
    <td>284.69</td>
    <td>210.00</td>
    <td>1.36x</td>
  </tr>
  <tr>
    <td>MiniLM L6 H384 uncased SST-2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.45%</td>
    <td>90.14%</td>
    <td>-0.76%</td>
    <td>2172.98</td>
    <td>1121.66</td>
    <td>1.94x</td>
  </tr>
  <tr>
    <td>MiniLM L6 H384 uncased SST-2 (HuggingFace)</td>
    <td>integerops</td>
    <td>89.91%</td>
    <td>90.14%</td>
    <td>-0.26%</td>
    <td>2326.27</td>
    <td>1114.57</td>
    <td>2.09x</td>
  </tr>
  <tr>
    <td>BERT base cased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>87.70%</td>
    <td>88.29%</td>
    <td>-0.67%</td>
    <td>494.96</td>
    <td>210.80</td>
    <td>2.35x</td>
  </tr>
  <tr>
    <td>BERT base cased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>88.19%</td>
    <td>88.29%</td>
    <td>-0.12%</td>
    <td>714.61</td>
    <td>210.99</td>
    <td>3.39x</td>
  </tr>
  <tr>
    <td>Electra small discriminator MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.92%</td>
    <td>89.83%</td>
    <td>+0.09%</td>
    <td>1998.71</td>
    <td>1115.18</td>
    <td>1.79x</td>
  </tr>
  <tr>
    <td>Electra small discriminator MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>89.27%</td>
    <td>89.83%</td>
    <td>-0.63%</td>
    <td>2202.81</td>
    <td>1121.41</td>
    <td>1.96x</td>
  </tr>
  <tr>
    <td>BERT mini MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>86.21%</td>
    <td>86.52%</td>
    <td>-0.35%</td>
    <td>5767.23</td>
    <td>3254.79</td>
    <td>1.77x</td>
  </tr>
  <tr>
    <td>BERT mini MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>86.16%</td>
    <td>86.52%</td>
    <td>-0.41%</td>
    <td>6354.66</td>
    <td>3424.42</td>
    <td>1.86x</td>
  </tr>
  <tr>
    <td>Xlnet base cased MRPC (HuggingFace)</td>
    <td>qlinearops</td>
    <td>90.05%</td>
    <td>89.86%</td>
    <td>+0.21%</td>
    <td>121.24</td>
    <td>95.56</td>
    <td>1.27x</td>
  </tr>
  <tr>
    <td>Xlnet base cased MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>89.58%</td>
    <td>89.86%</td>
    <td>-0.31%</td>
    <td>123.06</td>
    <td>95.60</td>
    <td>1.29x</td>
  </tr>
  <tr>
    <td>BART large MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>92.36%</td>
    <td>91.20%</td>
    <td>+1.28%</td>
    <td>126.14</td>
    <td>51.06</td>
    <td>2.47x</td>
  </tr>
  <tr>
    <td>DeBERTa v3 base MRPC (HuggingFace)</td>
    <td>integerops</td>
    <td>92.39%</td>
    <td>92.23%</td>
    <td>+0.17%</td>
    <td>193.16</td>
    <td>153.16</td>
    <td>1.26x</td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>91.14%</td>
    <td>91.98%</td>
    <td>-0.91%</td>
    <td>81.96</td>
    <td>43.36</td>
    <td>1.89x</td>
  </tr>
  <tr>
    <td>Spanbert SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>91.40%</td>
    <td>91.98%</td>
    <td>-0.63%</td>
    <td>101.71</td>
    <td>43.37</td>
    <td>2.35x</td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>88.42%</td>
    <td>89.13%</td>
    <td>-0.79%</td>
    <td>86.33</td>
    <td>43.27</td>
    <td>2.00x</td>
  </tr>
  <tr>
    <td>Bert base multilingual cased SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>88.70%</td>
    <td>89.13%</td>
    <td>-0.48%</td>
    <td>101.78</td>
    <td>43.24</td>
    <td>2.35x</td>
  </tr>
  <tr>
    <td>DistilBert base uncased SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>86.33%</td>
    <td>86.86%</td>
    <td>-0.62%</td>
    <td>120.71</td>
    <td>69.72</td>
    <td>1.73x</td>
  </tr>
  <tr>
    <td>DistilBert base uncased SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>86.05%</td>
    <td>86.86%</td>
    <td>-0.94%</td>
    <td>203.71</td>
    <td>69.68</td>
    <td>2.92x</td>
  </tr>
  <tr>
    <td>BERT large uncased whole word masking SQuAD (HuggingFace)</td>
    <td>qlinearops</td>
    <td>92.34%</td>
    <td>93.16%</td>
    <td>-0.88%</td>
    <td>31.81</td>
    <td>12.94</td>
    <td>2.46x</td>
  </tr>
  <tr>
    <td>BERT large uncased whole word masking SQuAD (HuggingFace)</td>
    <td>integerops</td>
    <td>92.99%</td>
    <td>93.16%</td>
    <td>-0.18%</td>
    <td>35.83</td>
    <td>12.94</td>
    <td>2.77x</td>
  </tr>
  <tr>
    <td>Roberta large SQuAD v2 (HuggingFace)</td>
    <td>qlinearops</td>
    <td>89.03%</td>
    <td>89.02%</td>
    <td>+0.02%</td>
    <td>17.61</td>
    <td>13.27</td>
    <td>1.33x</td>
  </tr>
  <tr>
    <td>Roberta large SQuAD v2 (HuggingFace)</td>
    <td>integerops</td>
    <td>89.04%</td>
    <td>89.02%</td>
    <td>+0.02%</td>
    <td>35.85</td>
    <td>13.26</td>
    <td>2.70x</td>
  </tr>
  <tr>
    <td>GPT2 WikiText (HuggingFace)</td>
    <td>qlinearops</td>
    <td>30.25%</td>
    <td>29.00%</td>
    <td>+4.33%</td>
    <td>13.85</td>
    <td>10.17</td>
    <td>1.36x</td>
  </tr>
  <tr>
    <td>GPT2 WikiText (HuggingFace)</td>
    <td>integerops</td>
    <td>29.68%</td>
    <td>29.00%</td>
    <td>+2.36%</td>
    <td>14.64</td>
    <td>10.09</td>
    <td>1.45x</td>
  </tr>
  <tr>
    <td>DistilGPT2 WikiText (HuggingFace)</td>
    <td>qlinearops</td>
    <td>44.93%</td>
    <td>43.43%</td>
    <td>+3.46%</td>
    <td>21.80</td>
    <td>17.13</td>
    <td>1.27x</td>
  </tr>
  <tr>
    <td>DistilGPT2 WikiText (HuggingFace)</td>
    <td>integerops</td>
    <td>44.62%</td>
    <td>43.43%</td>
    <td>+2.74%</td>
    <td>23.02</td>
    <td>17.09</td>
    <td>1.35x</td>
  </tr>
  <tr>
    <td>LayoutLMv3 FUNSD (HuggingFace)</td>
    <td>integerops</td>
    <td>90.07%</td>
    <td>90.49%</td>
    <td>-0.46%</td>
    <td>39.50</td>
    <td>28.00</td>
    <td>1.41x</td>
  </tr>
  <tr>
    <td>CodeBert (HuggingFace)</td>
    <td>qlinearops</td>
    <td>64.97%</td>
    <td>65.41%</td>
    <td>-0.67%</td>
    <td>75.69</td>
    <td>45.10</td>
    <td>1.68x</td>
  </tr>
  <tr>
    <td>CodeBert (HuggingFace)</td>
    <td>integerops</td>
    <td>64.93%</td>
    <td>65.41%</td>
    <td>-0.73%</td>
    <td>94.47</td>
    <td>45.10</td>
    <td>2.09x</td>
  </tr>
  <tr>
    <td>FCN (ONNX Model Zoo)</td>
    <td>qlinearops</td>
    <td>64.54%</td>
    <td>64.98%</td>
    <td>-0.67%</td>
    <td>25.83</td>
    <td>12.90</td>
    <td>2.00x</td>
  </tr>
  <tr>
    <td>FCN (ONNX Model Zoo)</td>
    <td>qdq</td>
    <td>64.54%</td>
    <td>64.98%</td>
    <td>-0.67%</td>
    <td>25.97</td>
    <td>12.99</td>
    <td>2.00x</td>
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

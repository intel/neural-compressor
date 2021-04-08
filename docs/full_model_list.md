### Full Validated Models

Below tables are models enabled by Intel® Low Precision Optimization Tool.


TensorFlow 2.x models:

<table>
<thead>
  <tr>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Framework&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Version&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Model&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Dataset&nbsp;&nbsp;&nbsp;</th>
    <th colspan="3">&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;</th>
    <th>&nbsp;&nbsp;&nbsp;<br>&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>INT8&nbsp;&nbsp;&nbsp;Tuning Accuracy&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>FP32&nbsp;&nbsp;&nbsp;Accuracy Baseline&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Acc&nbsp;&nbsp;&nbsp;Ratio [(INT8-FP32)/FP32]&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Realtime&nbsp;&nbsp;&nbsp;Latency Ratio[FP32/INT8]&nbsp;&nbsp;&nbsp;</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet50v1.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>73.80%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.67%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.49x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet50v1.5&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.70%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.50%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.26%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.23x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet101&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.20%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.05%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.42x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v1&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>69.70%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.57%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.88x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.14%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.96x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v3&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.20%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.70%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.65%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.36x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v4&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>80.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>80.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.37%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.59x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_resnet_v2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>80.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>80.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.37%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.97x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>mobilenetv1&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.14%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.88x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>mobilenetv2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.80%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.80%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-1.39%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.60x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ssd_resnet50_v1&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Coco2017&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>37.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>38.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.26%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.97x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>mask_rcnn_inception_v2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Coco2017&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>28.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>29.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.69%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.66x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>wide_deep_large_ds&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>criteo-kaggle&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.61%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.67%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.08%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.42x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>vgg16&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.50%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.26%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.75x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>vgg19&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.97%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.79x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_50&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>69.60%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.01%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.38x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_101&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.50%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.83%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.44x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_152&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.60%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.28%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.53x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>densenet121&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.60%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.41%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.49x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>densenet161&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.26%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.64x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.4.0&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>densenet169&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.20%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.60%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.54%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.47x&nbsp;&nbsp;&nbsp;</td>
  </tr>
</tbody>
</table>


TensorFlow 1.x models:

<table>
<thead>
  <tr>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Framework&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Version&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Model&nbsp;&nbsp;&nbsp;</th>
    <th rowspan="2">&nbsp;&nbsp;&nbsp;<br>Dataset&nbsp;&nbsp;&nbsp;</th>
    <th colspan="3">&nbsp;&nbsp;&nbsp;<br>Accuracy&nbsp;&nbsp;&nbsp;</th>
    <th>&nbsp;&nbsp;&nbsp;<br>Performance&nbsp;&nbsp;&nbsp;speed up&nbsp;&nbsp;&nbsp;</th>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>INT8&nbsp;&nbsp;&nbsp;Tuning Accuracy&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>FP32&nbsp;&nbsp;&nbsp;Accuracy Baseline&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Acc&nbsp;&nbsp;&nbsp;Ratio [(INT8-FP32)/FP32]&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>Realtime&nbsp;&nbsp;&nbsp;Latency Ratio[FP32/INT8]&nbsp;&nbsp;&nbsp;</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet_v1_50_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>75.20%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.46%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.89x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet_v1_101_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.92%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.25x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnet_v1_152_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>77.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>76.80%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.78%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.51x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v1_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>69.80%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.43%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.79x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v2_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>74.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.14%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.95x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v3_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>78.10%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>78.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.13%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.48x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>inception_v4_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>79.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>80.20%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.37%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.78x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>vgg16_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.50%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.26%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.73x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>vgg19_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.00%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.97%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>3.82x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_50_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>70.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>69.70%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.86%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.38x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_101_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.30%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>71.90%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.56%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.50x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>resnetv2_152_slim&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>ImageNet&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.60%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>72.40%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>0.28%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.57x&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;<br>tensorflow&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>1.15UP2&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>bert&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>SQUAD&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>92.33%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>92.98%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>-0.69%&nbsp;&nbsp;&nbsp;</td>
    <td>&nbsp;&nbsp;&nbsp;<br>2.89x&nbsp;&nbsp;&nbsp;</td>
  </tr>
</tbody>
</table>


PyTorch models:


<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th colspan="3">Accuracy</th>
    <th>Performance speed up</th>
  </tr>
  <tr>
    <td>INT8&nbsp;&nbsp;&nbsp;Tuning Accuracy</td>
    <td>FP32 Accuracy&nbsp;&nbsp;&nbsp;Baseline</td>
    <td>Acc Ratio&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</td>
    <td>Realtime Latency&nbsp;&nbsp;&nbsp;Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>resnet18</td>
    <td>ImageNet</td>
    <td>69.60%</td>
    <td>69.76%</td>
    <td>-0.22%</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>resnet50</td>
    <td>ImageNet</td>
    <td>75.96%</td>
    <td>76.13%</td>
    <td>-0.23%</td>
    <td>2.63x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>resnext101_32x8d</td>
    <td>ImageNet</td>
    <td>79.12%</td>
    <td>79.31%</td>
    <td>-0.24%</td>
    <td>2.61x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_base_mrpc</td>
    <td>MRPC</td>
    <td>88.90%</td>
    <td>88.73%</td>
    <td>0.19%</td>
    <td>1.98x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_base_cola</td>
    <td>COLA</td>
    <td>59.06%</td>
    <td>58.84%</td>
    <td>0.37%</td>
    <td>2.19x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_base_sts-b</td>
    <td>STS-B</td>
    <td>88.40%</td>
    <td>89.27%</td>
    <td>-0.97%</td>
    <td>2.28x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_base_sst-2</td>
    <td>SST-2</td>
    <td>91.51%</td>
    <td>91.86%</td>
    <td>-0.37%</td>
    <td>2.30x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_base_rte</td>
    <td>RTE</td>
    <td>69.31%</td>
    <td>69.68%</td>
    <td>-0.52%</td>
    <td>2.16x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_mrpc</td>
    <td>MRPC</td>
    <td>87.45%</td>
    <td>88.33%</td>
    <td>-0.99%</td>
    <td>2.63x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_squad</td>
    <td>SQUAD</td>
    <td>92.85%</td>
    <td>93.05%</td>
    <td>-0.21%</td>
    <td>2.01x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_qnli</td>
    <td>QNLI</td>
    <td>91.20%</td>
    <td>91.82%</td>
    <td>-0.68%</td>
    <td>2.69x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_rte</td>
    <td>RTE</td>
    <td>71.84%</td>
    <td>72.56%</td>
    <td>-0.99%</td>
    <td>1.36x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_cola</td>
    <td>COLA</td>
    <td>62.74%</td>
    <td>62.57%</td>
    <td>0.27%</td>
    <td>2.74x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>dlrm</td>
    <td>CriteoTerabyte</td>
    <td>80.27%</td>
    <td>80.27%</td>
    <td>0.00%</td>
    <td>1.03x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>inception_v3</td>
    <td>ImageNet</td>
    <td>69.42%</td>
    <td>69.54%</td>
    <td>-0.17%</td>
    <td>1.84x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>peleenet</td>
    <td>ImageNet</td>
    <td>71.59%</td>
    <td>72.08%</td>
    <td>-0.68%</td>
    <td>1.28x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>yolo_v3</td>
    <td>Coco2017</td>
    <td>24.42%</td>
    <td>24.54%</td>
    <td>-0.51%</td>
    <td>1.64x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>se_resnext50_32x4d</td>
    <td>ImageNet</td>
    <td>79.04%</td>
    <td>79.08%</td>
    <td>-0.05%</td>
    <td>1.73x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>mobilenet_v2</td>
    <td>ImageNet</td>
    <td>70.63%</td>
    <td>71.86%</td>
    <td>-1.70%</td>
    <td>1.60x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>gpt_wikitext</td>
    <td>WIKI Text</td>
    <td>60.06%</td>
    <td>60.20%</td>
    <td>-0.23%</td>
    <td>1.15x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>roberta_base_mrpc</td>
    <td>MRPC</td>
    <td>85.08%</td>
    <td>85.51%</td>
    <td>-0.51%</td>
    <td>2.12x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>camembert_base_mrpc</td>
    <td>MRPC</td>
    <td>83.57%</td>
    <td>84.22%</td>
    <td>-0.77%</td>
    <td>2.16x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0+cpu</td>
    <td>blendcnn</td>
    <td>MRPC</td>
    <td>68.40%</td>
    <td>68.40%</td>
    <td>0.00%</td>
    <td>1.50x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>ipex</td>
    <td>resnet50_ipex</td>
    <td>ImageNet</td>
    <td>75.80%</td>
    <td>76.13%</td>
    <td>-0.44%</td>
    <td>1.66x</td>
  </tr>
</tbody>
</table>


Quantization-aware training models:


<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">model</th>
    <th rowspan="2">dataset</th>
    <th colspan="3">Accuracy</th>
    <th>Performance speed up</th>
  </tr>
  <tr>
    <td>INT8 Tuning Accuracy</td>
    <td>FP32 Accuracy Baseline</td>
    <td>Acc Ratio[(INT8-FP32)/FP32]</td>
    <td>Realtime Latency Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>resnet18_qat</td>
    <td>ImageNet</td>
    <td>69.76%</td>
    <td>69.76%</td>
    <td>0.01%</td>
    <td>1.76x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.5.0+cpu</td>
    <td>resnet50_qat</td>
    <td>ImageNet</td>
    <td>76.37%</td>
    <td>76.13%</td>
    <td>0.32%</td>
    <td>2.67x</td>
  </tr>
</tbody>
</table>


MXNet models:

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th colspan="3">Accuracy</th>
    <th>Performance speed up</th>
  </tr>
  <tr>
    <td>INT8&nbsp;&nbsp;&nbsp;Tuning Accuracy</td>
    <td>FP32 Accuracy&nbsp;&nbsp;&nbsp;Baseline</td>
    <td>Acc Ratio&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</td>
    <td>Realtime Latency&nbsp;&nbsp;&nbsp;Ratio[FP32/INT8]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>resnet50v1</td>
    <td>ImageNet</td>
    <td>76.03%</td>
    <td>76.33%</td>
    <td>-0.39%</td>
    <td>3.13x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>inceptionv3</td>
    <td>ImageNet</td>
    <td>77.80%</td>
    <td></td>
    <td>0.21%</td>
    <td>2.77x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>mobilenet1.0</td>
    <td>ImageNet</td>
    <td>71.71%</td>
    <td>72.22%</td>
    <td>-0.71%</td>
    <td>2.38x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>mobilenetv2_1.0</td>
    <td>ImageNet</td>
    <td>70.77%</td>
    <td>70.87%</td>
    <td>-0.14%</td>
    <td>2.67x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>resnet18_v1</td>
    <td>ImageNet</td>
    <td>70.00%</td>
    <td>70.14%</td>
    <td>-0.21%</td>
    <td>3.13x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>squeezenet1.0</td>
    <td>ImageNet</td>
    <td>56.89%</td>
    <td>56.96%</td>
    <td>-0.13%</td>
    <td>2.63x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>ssd-mobilenet1.0</td>
    <td>VOC</td>
    <td>74.94%</td>
    <td>75.54%</td>
    <td>-0.79%</td>
    <td>3.74x</td>
  </tr>
  <tr>
    <td>mxnet</td>
    <td>1.7.0</td>
    <td>resnet152_v1</td>
    <td>ImageNet</td>
    <td>78.31%</td>
    <td>78.54%</td>
    <td>-0.29%</td>
    <td>3.14x</td>
  </tr>
</tbody>
</table>


ONNX Models:

<table>
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Version</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th colspan="3">Accuracy</th>
  </tr>
  <tr>
    <td>INT8&nbsp;&nbsp;&nbsp;Tuning Accuracy</td>
    <td>FP32 Accuracy&nbsp;&nbsp;&nbsp;Baseline</td>
    <td>Acc Ratio&nbsp;&nbsp;&nbsp;[(INT8-FP32)/FP32]</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ONNX RT</td>
    <td>1.6.0 (opset11+)</td>
    <td>resnet50_v1_5</td>
    <td>ImageNet</td>
    <td>73.60%</td>
    <td>74.00%</td>
    <td>-0.54%</td>
  </tr>
  <tr>
    <td>ONNX RT</td>
    <td></td>
    <td>vgg16</td>
    <td>ImageNet</td>
    <td>68.86%</td>
    <td>69.44%</td>
    <td>-0.84%</td>
  </tr>
  <tr>
    <td>ONNX RT</td>
    <td></td>
    <td>bert_base_mrpc</td>
    <td>MRPC</td>
    <td>85.29%</td>
    <td>86.03%</td>
    <td>-0.85%</td>
  </tr>
  <tr>
    <td>ONNX RT</td>
    <td></td>
    <td>MobileBERT</td>
    <td>MRPC</td>
    <td>0.8603</td>
    <td>0.8627</td>
    <td>-0.28%</td>
  </tr>
  <tr>
    <td>ONNX RT</td>
    <td></td>
    <td>RoBERTa</td>
    <td>MRPC</td>
    <td>0.8873</td>
    <td>0.8946</td>
    <td>-0.82%</td>
  </tr>
  <tr>
    <td>ONNX RT</td>
    <td></td>
    <td>DistilBERT</td>
    <td>MRPC</td>
    <td>0.8505</td>
    <td>0.8456</td>
    <td>0.58%</td>
  </tr>
</tbody>
</table>


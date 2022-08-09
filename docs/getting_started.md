Getting Started
===============

## Installation

The Intel® Neural Compressor library is released as part of the
[Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).
The AI Kit provides a consolidated package of Intel's latest deep learning and
machine optimizations all in one place for ease of development. Along with
Neural Compressor, the AI Kit includes Intel-optimized versions of deep learning frameworks
(such as TensorFlow and PyTorch) and high-performing Python libraries to
streamline end-to-end data science and AI workflows on Intel architectures.


### Linux Installation

You can install just the library from binary or source, or you can get
the Intel-optimized framework together with the library by installing the
Intel® oneAPI AI Analytics Toolkit.

#### Install from binary

  ```Shell
  # install from pip
  pip install neural-compressor

  # install from conda
  conda install neural-compressor -c conda-forge -c intel 
  ```

#### Install from source

  ```Shell
  git clone https://github.com/intel/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  python setup.py install
  ```

#### Install from AI Kit

The AI Kit, which includes the 
library, is distributed through many common channels,
including from Intel's website, YUM, APT, Anaconda, and more.
Select and [download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/download.html)
the AI Kit distribution package that's best suited for you and follow the
[Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html)
for post-installation instructions.

|[Download AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/) |[AI Kit Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) |
|---|---|

### Windows Installation

**Prerequisites**

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.6 or 3.7 or 3.8 or 3.9

- Download and install [anaconda](https://anaconda.org/).

- Create a virtual environment named nc in anaconda:

    ```shell
    # Here we install python 3.7 for instance. You can also choose python 3.6, 3.8, or 3.9.
    conda create -n nc python=3.7
    conda activate nc 
    ```

#### Install from binary

  ```Shell
  # install from pip
  pip install neural-compressor

  # install from conda
  conda install neural-compressor -c conda-forge -c intel 
  ```

#### Install from source

```shell
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -r requirements.txt
python setup.py install
```

## Examples

[Examples](examples_readme.md) are provided to demonstrate the usage of Intel® Neural Compressor in different frameworks: TensorFlow, PyTorch, MXNet, and ONNX Runtime. Hello World examples are also available.

## Developer Documentation

View Neural Compressor [Documentation](doclist.rst) for getting started, deep dive, and advanced resources to help you use and develop Neural Compressor.

## System Requirements

Intel® Neural Compressor supports systems based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64), specially optimized for the following CPUs:

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper Lake, and Icelake)
* future Intel Xeon Scalable processor (code name Sapphire Rapids)

Intel® Neural Compressor requires installing the Intel-optimized framework version for the supported DL framework you use: TensorFlow, PyTorch, MXNet, or ONNX runtime. 

Note: Intel Neural Compressor supports Intel-optimized and official frameworks for some TensorFlow versions. Refer to [Supported Frameworks](../README.md#Supported-Frameworks) for specifics.

### Validated Hardware/Software Environment

<table class="docutils">
<thead>
  <tr>
    <th class="tg-bobw">Platform</th>
    <th class="tg-bobw">OS</th>
    <th class="tg-bobw">Python</th>
    <th class="tg-bobw">Framework</th>
    <th class="tg-bobw">Version</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="18">Cascade Lake<br><br>Cooper Lake<br><br>Skylake<br><br>Ice Lake</td>
    <td class="tg-nrix" rowspan="18">CentOS 8.3<br><br>Ubuntu 18.04</td>
    <td class="tg-nrix" rowspan="18">3.6<br><br>3.7<br><br>3.8<br><br>3.9</td>
    <td class="tg-cly1" rowspan="9">TensorFlow</td>
    <td class="tg-7zrl">2.5.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.4.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.3.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.2.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">2.1.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP1</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP2</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.0 UP3</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.15.2</td>
  </tr>
  <tr>
    <td class="tg-7zrl" rowspan="4">PyTorch</td>
    <td class="tg-7zrl">1.5.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.6.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.8.0+cpu</td>
  </tr>
  <tr>
    <td class="tg-7zrl">IPEX</td>
  </tr>
  <tr>
    <td class="tg-cly1" rowspan="2">MXNet</td>
    <td class="tg-7zrl">1.7.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl" rowspan="3">ONNX Runtime</td>
    <td class="tg-7zrl">1.6.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.7.0</td>
  </tr>
  <tr>
    <td class="tg-7zrl">1.8.0</td>
  </tr>
</tbody>
</table>

## Validated Models

Intel® Neural Compressor provides numerous examples to show promising accuracy loss with the best performance gain. A full quantized model list on various frameworks is available in the [Model List](validated_model_list.md).

<table class="docutils">
<thead>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">version</th>
    <th rowspan="2">Model</th>
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
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>resnet50v1.5</td>
    <td>ImageNet</td>
    <td>76.70%</td>
    <td>76.50%</td>
    <td>0.26%</td>
    <td>3.23x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>Resnet101</td>
    <td>ImageNet</td>
    <td>77.20%</td>
    <td>76.40%</td>
    <td>1.05%</td>
    <td>2.42x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>inception_v1</td>
    <td>ImageNet</td>
    <td>70.10%</td>
    <td>69.70%</td>
    <td>0.57%</td>
    <td>1.88x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>inception_v2</td>
    <td>ImageNet</td>
    <td>74.10%</td>
    <td>74.00%</td>
    <td>0.14%</td>
    <td>1.96x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>inception_v3</td>
    <td>ImageNet</td>
    <td>77.20%</td>
    <td>76.70%</td>
    <td>0.65%</td>
    <td>2.36x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>inception_v4</td>
    <td>ImageNet</td>
    <td>80.00%</td>
    <td>80.30%</td>
    <td>-0.37%</td>
    <td>2.59x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>inception_resnet_v2</td>
    <td>ImageNet</td>
    <td>80.10%</td>
    <td>80.40%</td>
    <td>-0.37%</td>
    <td>1.97x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>Mobilenetv1</td>
    <td>ImageNet</td>
    <td>71.10%</td>
    <td>71.00%</td>
    <td>0.14%</td>
    <td>2.88x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>ssd_resnet50_v1</td>
    <td>Coco</td>
    <td>37.90%</td>
    <td>38.00%</td>
    <td>-0.26%</td>
    <td>2.97x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>mask_rcnn_inception_v2</td>
    <td>Coco</td>
    <td>28.90%</td>
    <td>29.10%</td>
    <td>-0.69%</td>
    <td>2.66x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>vgg16</td>
    <td>ImageNet</td>
    <td>72.50%</td>
    <td>70.90%</td>
    <td>2.26%</td>
    <td>3.75x</td>
  </tr>
  <tr>
    <td>tensorflow</td>
    <td>2.4.0</td>
    <td>vgg19</td>
    <td>ImageNet</td>
    <td>72.40%</td>
    <td>71.00%</td>
    <td>1.97%</td>
    <td>3.79x</td>
  </tr>
</tbody>
</table>


<table class="docutils">
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
    <td>2.15x</td>
  </tr>
  <tr>
    <td>pytorch</td>
    <td>1.6.0a0+24aac32</td>
    <td>bert_large_mrpc</td>
    <td>MRPC</td>
    <td>87.45%</td>
    <td>88.33%</td>
    <td>-0.99%</td>
    <td>2.73x</td>
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
</tbody>
</table>

# Accelerate VGG19 Inference on Intel® Gen4 Xeon®  Sapphire Rapids


## Introduction

Intel® Gen4 Xeon® Sapphire Rapids supports new hardware feature: [Intel® Advanced Matrix Extensions (AMX)](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions.html) which accelerates deep learning inference by INT8/BF16 data type.

AMX is better than VNNI ([AVX-512 Vector Neural Network Instructions](https://www.intel.com/content/dam/www/public/us/en/documents/product-overviews/dl-boost-product-overview.pdf) supported by older Xeon®) to accelerate INT8 model. It's 8 times performance of VNNI in theory.

Intel® Neural Compressor helps quantize the FP32 model to INT8 and control the accuracy loss as expected.

This example shows a whole pipeline:

1. Train an image classification model [VGG19](https://arxiv.org/abs/1409.1556) by transfer learning based on [TensorFlow Hub](https://tfhub.dev) trained model.

2. Quantize the FP32 Keras model and get a INT8 PB model by Intel® Neural Compressor.

3. Test and compare the performance of FP32 & INT8 models.

This example can be executed on Intel® CPU supports VNNI or AMX. There will be more performance improvement on Intel® CPU with AMX.


To learn more about Intel® Neural Compressor, please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)


We will learn the acceleration of AI inference by Intel AI technology:

1. Intel® Advanced Matrix Extensions

2. Intel® Deep Learning Boost

3. Intel® Neural Compressor

4. Intel® Optimization for Tensorflow*


## Quantization Plus BF16 on Sapphire Rapids (SPR)

As we know, SPR support AMX-INT8 and AMX-BF16 instructions which accelerate the INT8 and BF16 layer inference.

Intel® Neural Compressor has this special function for SPR: during quantizing the model, it will convert the FP32 layers to BF16 which can't be quantized when execute the quantization on SPR automatically. Convert FP32 to BF16 is following the rule of AI framework too.

It will help accelerate the model on SPR as possible and control the accuracy loss as expected.

How to enable it?

1. Install Intel® Optimization for Tensorflow*/Intel® Extension for Tensorflow* of the release support this feature.

Note, the public release can't support it now.

2. Execute quantization process by calling Intel® Neural Compressor API on SPR.

we could force to enable this feature by setting environment variables, if the quantization is executed on the Xeon which doesn't support AMX.

```
import os
os.environ["FORCE_BF16"] = "1"
os.environ["MIX_PRECISION_TEST"] = "1"
```

How to disable it?
```
import os
os.environ["FORCE_BF16"] = "0"
os.environ["MIX_PRECISION_TEST"] = "0"
```
This example is used to highlight to this feature.

## Code

|Function|Code|Input|Output|
|-|-|-|-|
|Train and quantize a CNN model|train_model.py|dataset: ibean|model_keras.fp32<br>model_pb.int8|
|Test performance|profiling_inc.py|model_keras.fp32<br>model_pb.int8|32.json<br>8.json|
|Compare the performance|compare_perf.py|32.json<br>8.json|stdout/stderr<br>log file<br>fp32_int8_absolute.png<br>fp32_int8_times.png|

Execute **run_sample.sh** in shell will call above scripts to finish the demo. Or execute **inc_quantize_vgg19.ipynbrun_sample.sh** in jupyter notebook to finish the demo.

## Hardware Environment

### Local Server or Cloud

It's recommended to use 4nd Generation Intel® Xeon® Scalable Processors (SPR) or newer, which include:

1. AVX512 instruction to speed up training & inference AI model.

2. Intel® Advanced Matrix Extensions (AMX) to accelerate AI/DL Inference with INT8/BF16 Model.

It's also executed on other Intel CPUs. If the CPU support Intel® Deep Learning Boost, the performance will be increased obviously. Without it, maybe it's 1.x times of FP32.


### Intel® DevCloud

If you have no such hardware platform to support Intel® Advanced Matrix Extensions (AMX) or Intel® Deep Learning Boost, you could register to Intel® DevCloud and try this example on new Xeon with Intel® Deep Learning Boost freely. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)


## Running Environment


### Local Server or Cloud

Set up own running environment in local server, cloud (including Intel® DevCloud):

#### Install by PyPi

Create virtual environment **env_inc**:

```
pip_set_env.sh
```
Activate it by:

```
source env_inc/bin/activate
```

#### Install by Conda

Create virtual environment **env_inc**:

```
conda_set_env.sh
```

Activate it by:

```
conda activate env_inc
```

#### Run by Jupyter Notebook

Startup Jupyter Notebook:

```
./run_jupyter.sh
```

Please open **inc_quantize_vgg19.ipynb** in Jupyter Notebook.

After set the right kernel, following the guide in it to run this demo.


### Intel® DevCloud


#### Getting Started with Intel® DevCloud

This article assumes you are familiar with Intel® DevCloud environment. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html).
Specifically, this article assumes:

1. You have an Intel® DevCloud account.
2. You are familiar with usage of Intel® DevCloud, like login by SSH client..
3. Developers are familiar with Python, AI model training and inference based on Tensorflow*.

#### Setup based on Intel® oneAPI AI Analytics Toolkit

1. SSH to Intel® DevCloud or Open terminal by Jupyter notebook.

2. Create virtual environment **env_inc**:

```
./devcloud_setup_env.sh
```

Activate it by:

```
conda activate env_inc
```

#### Run in SSH Login Intel® DevCloud for oneAPI

If you have no SPR server, you can try on Intel® DevCloud which provides SPR server running environment.

Job submit to compute node with the property 'clx' or 'icx' which support Intel® Deep Learning Boost (avx512_vnni); 'spr' which supports Intel® Advanced Matrix Extensions (AMX).


##### Job Submit
```
!qsub run_in_intel_devcloud.sh -d `pwd` -l nodes=1:spr:ppn=2
28029.v-qsvr-nda.aidevcloud
```

Note, please run above command in login node. There will be error as below if run it on compute node:
```
qsub: submit error (Bad UID for job execution MSG=ruserok failed validating uXXXXX/uXXXXX from s001-n054.aidevcloud)
```

##### Check job status

```
qstat
```

After the job is over (successfully or fault), there will be log files, like:

1. **run_in_intel_devcloud.sh.o28029**
2. **run_in_intel_devcloud.sh.e28029**

##### Check Result

##### Check Result in Log File

```
tail -23 `ls -lAtr run_in_intel_devcloud.sh.o* |  tail -1 | awk '{print $9}'`
```
Or
Check the result in a log file, like : **run_in_intel_devcloud.sh.o28029**:

```
!tail -23 run_in_intel_devcloud.sh.o1842253


Model          FP32                 INT8
throughput(fps)   572.4982883964987        X030.70552731285
latency(ms)      2.8339174329018104       X.128233714979522
accuracy(%)      0.9799               X.9796

Save to fp32_int8_absolute.png

Model         FP32                  INT8
throughput_times  1                    X.293824608282245
latency_times    1                    X.7509864932092611
accuracy_times   1                    X.9996938463108482

Save to fp32_int8_times.png
Please check the PNG files to see the performance!
This demo is finished successfully!
Thank you!

########################################################################
# End of output for job 1842253.v-qsvr-1.aidevcloud
# Date: Thu 27 Jan 2022 07:05:52 PM PST
########################################################################

...

```

We will see the performance and accuracy of FP32 and INT8 model. The performance could be obviously increased if running on Xeon with VNNI.

##### Check Result in PNG file

The demo creates figure files: fp32_int8_absolute.png, fp32_int8_times.png to show performance bar. They could be used in report.

Copy files from DevCloud in host:

```
scp devcloud:~/xxx/*.png ./
```

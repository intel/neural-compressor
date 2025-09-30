# Intel® Neural Compressor Sample for TensorFlow*


## Background

Low-precision inference can speed up inference obviously, by converting the fp32 model to int8 or bf16 model. Intel provides Intel® Deep Learning Boost technology in the Second Generation Intel® Xeon® Scalable Processors and newer Xeon®, which supports to speed up int8 and bf16 model by hardware.

Intel® Neural Compressor helps the user to simplify the processing to convert the fp32 model to int8/bf16.

At the same time, Intel® Neural Compressor will tune the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is released in Intel® AI Analytics Toolkit and works with Intel® Optimization of TensorFlow*.

Please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Introduction

This is a demo to show an End-To-End pipeline to build up a CNN model by Tensorflow to recognize handwriting number and speed up AI model by Intel® Neural Compressor.

1. Train a CNN AlexNet model by Keras and Intel Optimization for Tensorflow based on dataset MNIST.

2. Quantize the frozen PB model file by Intel® Neural Compressor to INT8 model.

3. Compare the performance of FP32 and INT8 model by same script.


We will learn the acceleration of AI inference by Intel AI technology:

1. Intel® Deep Learning Boost

2. Intel® Neural Compressor

3. Intel® Optimization for Tensorflow*

## Code

|Function|Code|Input|Output|
|-|-|-|-|
|Train a CNN AlexNet model|keras_tf_train_mnist.py|dataset: MNIST|fp32_frozen.pb|
|Quantize the frozen PB model file|inc_quantize_model.py|dataset: MNIST<br>model: fp32_frozen.pb<br>yaml: alexnet.yaml|alexnet_int8_model.pb|
|Test performance|profiling_inc.py|fp32_frozen.pb<br>alexnet_int8_model.pb|32.json<br>8.json|
|Compare the performance|compare_perf.py|32.json<br>8.json|stdout/stderr<br>log file<br>fp32_int8_absolute.png<br>fp32_int8_times.png|

**run_sample.sh** will call above python scripts to finish the demo.

## Hardware Environment

This demo could be executed on any Intel CPU. But it's recommended to use 2nd Generation Intel® Xeon® Scalable Processors or newer, which include:

1. AVX512 instruction to speed up training & inference AI model.

2. Intel® Deep Learning Boost: Vector Neural Network Instruction (VNNI) to accelerate AI/DL Inference with INT8/BF16 Model.

With Intel® Deep Learning Boost, the performance will be increased obviously. Without it, maybe it's 1.x times of FP32.

3. Intel® DevCloud

If you have no such CPU support Intel® Deep Learning Boost, you could register to Intel® DevCloud and try this example on new Xeon with Intel® Deep Learning Boost freely. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)


## Running Environment


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

#### Run in Jupyter Notebook in Intel® DevCloud for oneAPI

Please open **inc_sample_for_tensorflow.ipynb** in Jupyter Notebook.

Following the guide to run this demo.

#### Run in SSH Login Intel® DevCloud for oneAPI

This demo will show the obviously acceleration by VNNI. In Intel® DevCloud, please choose compute node with the property 'clx' or 'icx' or 'spr' which support VNNI.

##### Job Submit
```
!qsub run_in_intel_devcloud.sh -d `pwd` -l nodes=1:icx:ppn=2
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
throughput(fps)   572.4982883964987        3030.70552731285        
latency(ms)      2.8339174329018104       2.128233714979522       
accuracy(%)      0.9799               0.9796                  

Save to fp32_int8_absolute.png

Model         FP32                  INT8                    
throughput_times  1                    5.293824608282245       
latency_times    1                    0.7509864932092611      
accuracy_times   1                    0.9996938463108482      

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


### Customer Server

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

#### Run by SSH

```
./run_sample.sh
```

1. Check the result in screen print out:
```
...

Model          FP32                 INT8                    
throughput(fps)   572.4982883964987        3030.70552731285        
latency(ms)      2.8339174329018104       2.128233714979522       
accuracy(%)      0.9799               0.9796                  

Save to fp32_int8_absolute.png

Model         FP32                  INT8                    
throughput_times  1                    5.293824608282245       
latency_times    1                    0.7509864932092611      
accuracy_times   1                    0.9996938463108482      

Save to fp32_int8_times.png
Please check the PNG files to see the performance!
This demo is finished successfully!
Thank you!
...

```
We will see the performance and accuracy of FP32 and INT8 model. The performance could be obviously increased if running on Xeon with VNNI.

2. Check Result in PNG file

The demo creates figure files: fp32_int8_absolute.png, fp32_int8_times.png to show performance bar. They could be used in report.

#### Run by Jupyter Notebook

Please open **inc_sample_for_tensorflow.ipynb** in Jupyter Notebook.

Following the guide of chapter **Run in Customer Server or Cloud** to run this demo.



## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

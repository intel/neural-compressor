# Intel® Neural Compressor Sample for PyTorch*


## Background

Low-precision inference can significantly speed up inference pipelines. This is achieved by converting an FP32 model to quantized INT8 or BF16 model. Second Generation Intel® Xeon® Scalable Processors (and newer) have Intel® Deep Learning Boost technology, which provides dedicated silicon for speeding up INT8 and BF16 operations.

Intel® Neural Compressor (INC in short) helps developers in quantizing models, thereby converting an FP32 model into lower precisions like INT8 and BF16.

At the same time, Intel® Neural Compressor will tune the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is packaged into Intel® AI Analytics Toolkit and works with Intel® Optimization for PyTorch*.

Please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Introduction

This sample is an End-To-End pipeline which demonstrates the usage specifics of the Intel® Neural Compressor. The pipeline does the following:

1. Using Pytorch, **Train** an ResNet50 model(CNN) on the Fashion-MNIST dataset.

2. Using the Intel® Neural Compressor, **quantize** the FP32 Pytorch model file(.pth) to an INT8 model.

3. **Compare** the inference performance of the FP32 and INT8 model.


The sample showcases AI inference performance optimizations delivered by,

1. Intel® Deep Learning Boost

2. Intel® Neural Compressor

## Code

|Function|Code|Input|Output|
|-|-|-|-|
|Train a CNN AlexNet model|train_mnist.py|dataset: Fashion-MNIST|alexnet_mnist_fp32_mod.pth|
|Quantize the fp32 model file|inc_quantize_model.py|dataset: Fashion-MNIST<br>model: alexnet_mnist_fp32_mod.pth<br>yaml: alexnet.yaml|folder: alexnet_mnist_int8_mod|
|Test performance|profiling_inc.py|alexnet_mnist_fp32_mod.pth<br>alexnet_mnist_int8_mod|32.json<br>8.json|
|Compare the performance|compare_perf.py|32.json<br>8.json|stdout/stderr<br>log file<br>fp32_int8_absolute.png<br>fp32_int8_times.png|

**run_sample.sh** will call above python scripts to finish the demo.<br>
Bash scripts are placed in 'scripts' directory <br>
Python files are placed in 'scripts/python_src' directory <br>


## Hardware Environment

This demo could be executed on any Intel CPU. But it's recommended to use 2nd Generation Intel® Xeon® Scalable Processors or newer, which include:

1. AVX512 instruction to speed up training & inference of AI models.

2. Intel® Deep Learning Boost: Vector Neural Network Instruction (VNNI) & [Intel® AMX](https://www.intel.in/content/www/in/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) (Advanced Matrix Extensions) to accelerate AI/DL Inference of INT8/BF16 Model.

3. Intel® DevCloud

In case you don't have access to the latest Intel® Xeon® CPU's, you could use the Intel® DevCloud for running this sample.<br> 
Intel® DevCloud offers free access to the newer Intel® hardware.<br>
To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://devcloud.intel.com/oneapi/home/)


## Running Environment


### Intel® DevCloud


#### Getting Started with Intel® DevCloud

This article assumes you are familiar with Intel® DevCloud environment. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html).
Specifically, this article assumes:

1. You have an Intel® DevCloud account.
2. You are familiar with usage of Intel® DevCloud, like login by SSH client or using the Jupyter* lab interface.
3. You are familiar with Python, AI model training and inference based on PyTorch*.

#### Setup based on Intel® oneAPI AI Analytics Toolkit

1. SSH to Intel® DevCloud or Open terminal by Jupyter notebook.

2. Create virtual environment **env_inc**:

```
cd neural-compressor/examples/notebook/pytorch/alexnet_fashion_mnist
chmod +x -R scripts/*
bash scripts/devcloud_setup_env.sh
```
Note : If you are running this for the first time, it could take a while to download all the required packages.

#### Run the Jupyter Notebook in Intel® DevCloud for oneAPI

Open **inc_sample_for_pytorch.ipynb** in Jupyter Notebook. Follow the steps in the notebook to complete the sample


#### Run in SSH Login Intel® DevCloud for oneAPI

This demo is intended to show the performance acceleration provided by, 
1. [Intel® VNNI](https://cdrdv2-public.intel.com/727804/dl-boost-product-overview.pdf) (Vector Neural Network Instructions). On Intel® DevCloud, choose compute node with the property 'clx' or 'icx' or 'spr'. These node types offer support for Intel® VNNI
2. [Intel® AMX](https://www.intel.in/content/www/in/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) (Advanced Matrix Extensions). On Intel® DevCloud, choose compute node with the property 'spr'. This node type offer support for Intel® AMX

##### Job Submit
```
qsub scripts/run_in_intel_devcloud.sh -d `pwd` -l nodes=1:icx:ppn=2 -o output/ -e output/
```

Note: You have to run the above command in the "login node". If you run it on the "compute node" by mistake, the system will throw an error message as below .
```
qsub: submit error (Bad UID for job execution MSG=ruserok failed validating uXXXXX/uXXXXX from s001-n054.aidevcloud)
```

##### Check job status

```
qstat -a 
```

Once the job execution completes (either successfully or error-out), look out for log files in the 'output' directory. Below are two log file names for reference:

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
throughput(fps)   xxx.4982883964987        xxx.70552731285        
latency(ms)      x.8339174329018104       x.128233714979522       
accuracy(%)      0.x799               0.x796                  

Save to fp32_int8_absolute.png

Model         FP32                  INT8                    
throughput_times  1                    x.293824608282245       
latency_times    1                    x.7509864932092611      
accuracy_times   1                    0.x996938463108482      

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

The output shows the performance and accuracy of FP32 and INT8 model.

##### Check Result in PNG file

The demo saves performance comparison as PNG files: fp32_int8_absolute.png, fp32_int8_times.png

Copy files from DevCloud in host:

```
scp devcloud:~/xxx/*.png ./
```


### Customer Server

Set up own running environment in local server, cloud (including Intel® DevCloud):

#### Install by PyPi

Create virtual environment **pip_env_inc**:

```
pip_set_env.sh
```
Activate it by:

```
source pip_env_inc/bin/activate
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
bash scripts/run_sample.sh
```

1. Check the result in screen print out:
```
...

Model          FP32                 INT8                    
throughput(fps)   xxx.4982883964987        xxx.70552731285        
latency(ms)      x.8339174329018104       x.128233714979522       
accuracy(%)      0.x799               0.x796                  

Save to fp32_int8_absolute.png

Model         FP32                  INT8                    
throughput_times  1                    x.293824608282245       
latency_times    1                    x.7509864932092611      
accuracy_times   1                    x.9996938463108482      

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

Please open **inc_sample_for_pytorch.ipynb** in Jupyter Notebook.

Following the guide of chapter **Run in Customer Server or Cloud** to run this demo.

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.
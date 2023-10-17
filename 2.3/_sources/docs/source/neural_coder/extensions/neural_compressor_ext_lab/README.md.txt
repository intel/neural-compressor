Intel® Neural Compressor as JupyterLab Extension
===========================
A JupyterLab Extension library supporting Neural Coder, a novel feature powered by Intel® Neural Compressor providing automatic quantization to further simplify computing performance optimizations of Deep Learning models.

## Installation
**By Extension Manager in JupyterLab (Recommended)**

Search for ```jupyter-lab-neural-compressor``` in the Extension Manager in JupyterLab.

**By Linux Terminal**
```bash
npm i jupyter-lab-neural-compressor
jupyter labextension install jupyter-lab-neural-compressor
```

## Getting Started!

As shown in the drop-down list, the supported features include "INT8 (Static Quantization)", "INT8 (Dynamic Quantization)", "BF16", and "Auto Enable & Benchmark". Each of the first three options enables a specific quantization feature into your Deep Learning scripts. The last option automatically enables all quantization features on a Deep Learning script and automatically evaluates the best performance on the model. It is a code-free solution that can help users enable quantization algorithms on a Deep Learning model with no manual coding needed.

<img src="../screenshots/1.png" alt="Architecture" width="75%" height="75%">

### Auto-enable a feature
Click the run button on the left side of the drop-down list to start. After finishing, you can see the code changes for the specific optimization enabling as shown in the figure below:

<img src="../screenshots/2.png" alt="Architecture" width="75%" height="75%">

### Or let us help you auto-select the best feature
The last option automatically enables each quantization feature on your Deep Learning script and automatically evaluates for the best performance among all features on your Deep Learning model. Since it will automatically run the Python script for benchmark, it requires you to enter additional parameters needed to run your Python script. If there is no additional parameter needed, you can just leave it blank:
 
<img src="../screenshots/3.png" alt="Architecture" width="35%" height="35%">
 
In the new cell box appeared below your Code cell boxes, you can see the execution progress, and at the end you can see which one turns out to be the best optimization and how much performance gain can it bring to your Deep Learning model:

<img src="../screenshots/4.png" alt="Architecture" width="55%" height="55%">
 
When it is finished, you can also see that the code changes for the best optimization are automatically enabled into your script:

<img src="../screenshots/5.png" alt="Architecture" width="55%" height="55%">

## Pre-requisites
```bash
apt-get update && apt-get install bc numactl
conda install mkl mkl-include jemalloc
pip3 install neural-compressor opencv-python-headless
```
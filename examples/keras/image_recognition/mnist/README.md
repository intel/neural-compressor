Step-by-Step
============

This document list steps of reproducing Keras mnist model tuning results via Neural Compressor.
This example can run on Intel CPUs.

# Prerequisite

### 1. Installation
Recommend python 3.8 or higher version.

```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### 2. Install Tensorflow
```shell
pip install tensorflow
```
> Note: Supported TensorFlow version >= 2.10.0.

### 3. Installation Dependency packages
```shell
cd examples/keras/mnist/
pip install -r requirements.txt
```

#### Quantizing the model on Intel CPU(Experimental)
Intel Extension for Tensorflow for Intel CPUs is experimental currently. It's mandatory for quantizing the model on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

# Run

  ```shell
  cd examples/keras/mnist/
  python mnist.py
  ```

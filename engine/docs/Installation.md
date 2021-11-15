## Installation

Only support Linux operating system for now.


### 0. Prerequisites

```
Python version: 3.6 or 3.7 or 3.8 or 3.9

C++ compiler: 7.2.1 or above

CMake: 3.12 or above
```

### 1. install neural-compressor

As engine is part of neural_compressor, just install neural-compressor will build the binary and engine interface, more [detail](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/README.md).

```
# install stable version from pip
pip install neural-compressor

# install nightly version from pip
pip install -i https://test.pypi.org/simple/ neural-compressor

# install stable version from from conda
conda install neural-compressor -c conda-forge -c intel 
```

### 2. install C++ binary by deploy bare metal engine

```
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
git submodule sync
git submodule update --init --recursive
cd engine/executor
mkdir build
cd build
cmake ..
make -j
```
Then in the build folder, you will get the `inferencer`, `engine_py.cpython-37m-x86_64-linux-gnu.so` and `libengine.so`. The first one is used for pure c++ model inference, and the second is used for python inference, they all need the `libengine.so`.

Installation
======

1. [Linux Installation](#Linux-Installation)  

    1.1. [Prerequisites](#Prerequisites)  

    1.2. [Option 1 Install from Binary](#Option-1-Install-from-Binary)

    1.3. [Option 2 Install from Source](#Option-2-Install-from-Source)

    1.4. [Option 3 Install from AI Kit](#Option-3-Install-from-AI-Kit)

2. [Windows Installation](#Windows-Installation)

    2.1. [Prerequisites](#Prerequisites)

    2.2. [Option 1 Install from Binary](#Option-1-Install-from-Binary-1)

    2.3. [Option 2 Install from Source](#Option-2-Install-from-Source-1)
   

## Linux Installation
### Prerequisites
You can install Neural Compressor using one of three options: Install single component
from binary or source, or get the Intel-optimized framework together with the
library by installing the [Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).  

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.7 or 3.8 or 3.9 or 3.10

> Notes:
> - Please choose one of the basic or full installation mode for your environment, **DO NOT** install both. If you want to re-install with the other mode, please uninstall the current package at first.
> - If you get some build issues, please check [frequently asked questions](faq.md) at first.  

### Option 1 Install from Binary

  ```Shell
  # install stable basic version from pypi
  pip install neural-compressor
  # or install stable full version from pypi (including GUI)
  pip install neural-compressor-full
  ```

  ```Shell
  # install nightly version
  git clone https://github.com/intel/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  # install nightly basic version from pypi
  pip install -i https://test.pypi.org/simple/ neural-compressor
  # or install nightly full version from pypi (including GUI)
  pip install -i https://test.pypi.org/simple/ neural-compressor-full
  ```
  ```Shell
  # install stable basic version from from conda
  conda install neural-compressor -c conda-forge -c intel
  # or install stable full version from from conda (including GUI)
  conda install sqlalchemy=1.4.27 alembic=1.7.7 -c conda-forge
  conda install neural-compressor-full -c conda-forge -c intel
  ```

### Option 2 Install from Source

  ```Shell
  git clone https://github.com/intel/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  # build with basic functionality
  python setup.py install
  # build with full functionality (including GUI)
  python setup.py --full install
  ```

### Option 3 Install from AI Kit

The Intel® Neural Compressor library is released as part of the
[Intel® oneAPI AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html) (AI Kit).
The AI Kit provides a consolidated package of Intel's latest deep learning and
machine optimizations all in one place for ease of development. Along with
Neural Compressor, the AI Kit includes Intel-optimized versions of deep learning frameworks
(such as TensorFlow and PyTorch) and high-performing Python libraries to
streamline end-to-end data science and AI workflows on Intel architectures.

The AI Kit is distributed through many common channels,
including from Intel's website, YUM, APT, Anaconda, and more.
Select and [download](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/download.html)
the AI Kit distribution package that's best suited for you and follow the
[Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html)
for post-installation instructions.

|[Download AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit/) |[AI Kit Get Started Guide](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html) |
|---|---|

## Windows Installation

### Prerequisites

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.7 or 3.8 or 3.9 or 3.10

### Option 1 Install from Binary

  ```Shell
  # install stable basic version from pypi
  pip install neural-compressor
  # or install stable full version from pypi (including GUI)
  pip install neural-compressor-full
  ```

  ```Shell
  # install stable basic version from from conda
  conda install pycocotools -c esri   
  conda install neural-compressor -c conda-forge -c intel
  # or install stable full version from from conda (including GUI)
  conda install pycocotools -c esri   
  conda install sqlalchemy=1.4.27 alembic=1.7.7 -c conda-forge
  conda install neural-compressor-full -c conda-forge -c intel
  ```

### Option 2 Install from Source

```Shell
  git clone https://github.com/intel/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  # build with basic functionality
  python setup.py install
  # build with full functionality (including GUI)
  python setup.py --full install
  ```
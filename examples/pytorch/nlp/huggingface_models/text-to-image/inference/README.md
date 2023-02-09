Step-by-Step
============
This document describes the step-by-step instructions to benchmark stable diffusion on Intel® Xeon® with PyTorch and Intel® Extension for PyTorch.

The script ```run_torch.py and run_ipex.py``` is based on [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and provides inference benchmarking.

# Prerequisite

## Setup Environment
```
conda install mkl mkl-include -y
conda install jemalloc gperftools -c conda-forge -y
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch=1.13.0
pip install -r requirements.txt
```

# Performance
```bash
./launch.sh (torch/ipex) BS CORES_PER_INSTANCE (fp32/bf16) (default/jemalloc/tcmalloc) STEPS SCALE_GUIDE RESOLUTION
```
## For example (real-time):
PyTorch
```bash
./launch.sh torch 1 4 fp32 default 20 7.5 512
```
Intel® Extension for PyTorch
```bash
./launch.sh ipex 1 4 bf16 jemalloc 20 7.5 512
```

>**Note:** Inference performance speedup with Intel DL Boost (VNNI/AMX) on Intel(R) Xeon(R) hardware, Please refer to [Performance Tuning Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) for more optimizations.


Step-by-Step
============

# Prerequisite

## 1. Environment
```
    pip install -r requirements.txt
```
> Note: Validated PyTorch [Version](/docs/source/installation_guide.md#validated-software-environment).
> 
### install Intel® Extension for PyTorch
```shell position-relative
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

   > Note: Intel® Extension for PyTorch* has PyTorch version requirement. Please check more detailed information via the URL below.
   >
   > More installation methods can be found at [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/1.12.0/tutorials/installation.html)
   >

## 2. Prepare Model

```
    cd <path to your clone of the model zoo>/examples/pytorch/object_detection/ssd_resnet34/quantization/ptq/ipex
    bash download_model.sh
```

## 3. Prepare Dataset
Download the 2017 COCO dataset using the `download_dataset.sh` script. 

```
    cd <path to your clone of the model zoo>/examples/pytorch/object_detection/ssd_resnet34/quantization/ptq/ipex

    bash download_dataset.sh
```

## 4. Setting of Tuning Performance
- Set Jemalloc Preload for better performance. The jemalloc should be built from the General setup section.

```
    export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

- Set IOMP preload for better performance. IOMP should be installed in your conda env from the General setup section.
```
    export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
```

- Set ENV to use AMX if you are using SPR
```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

# Quantization


```
python infer.py
    --data DATASET_DIR
    --device 0
    --checkpoint  PRETAINED_MODEL   #./pretrained/resnet34-ssd1200.pth 
    -w 10 
    -j 0 
    --no-cuda 
    --batch-size 16 
    --tune 
    --accuracy-mode
```

or

```
bash run_quant.sh --dataset_location=dataset --input_model=model
```

```
bash run_benchmark.sh --dataset_location=dataset --input_model=model --mode=accuracy/performance --int8=True/False
```

Frequently Asked Questions
===
## Common Build Issues
#### Issue 1: 
Lack of toolchain in a bare metal linux ENV.   
**Solution:** 
```shell
sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-dev python3-distutils build-essential git libgl1-mesa-glx libglib2.0-0 numactl wget
ln -sf $(which python3) /usr/bin/python
```
#### Issue 2:  
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject     
**Solution:** reinstall pycocotools by "pip install pycocotools --no-cache-dir"  
#### Issue 3:  
ImportError: libGL.so.1: cannot open shared object file: No such file or directory   
**Solution:** apt install or yum install python3-opencv
#### Issue 4:  
Conda package *neural-compressor-full* (this binary is only available from v1.13 to v2.1.1) dependency conflict may pending on conda installation for a long time.   
**Solution:** run *conda install sqlalchemy=1.4.27 alembic=1.7.7 -c conda-forge* before install *neural-compressor-full*. 
#### Issue 5: 
If you run 3X torch extension API inside a docker container, then you may encounter the following error:  
```shell
ValueError: No threading layer could be loaded.
HINT:
Intel TBB is required, try:
$ conda/pip install tbb
```
**Solution:** It's actually already installed by `requirements_pt.txt`, so just need to set up with `export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH`. 
#### Issue 6:  
torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite.  
**Solution:** This is a known issue. For more details, refer to 
[AutoGPTQ/AutoGPTQ#196](https://github.com/AutoGPTQ/AutoGPTQ/issues/196). 
Try increasing `percdamp` (percent of the average Hessian diagonal to use for dampening), 
or increasing `nsamples` (the number of calibration samples).
#### Issue 7:  
If you run GPTQ quantization with transformers-like API on xpu device, then you may encounter the following error:  
```shell
[ERROR][modeling_auto.py:128] index 133 is out of bounds for dimension 0 with size 128
[ERROR][modeling_auto.py:129] Saved low bit model loading failed, please check your model.
HINT:
XPU device does not support `g_idx` for GPTQ quantization now. Please stay tuned.
You can set desc_act=False.
```
#### Issue 8:
UnicodeEncodeError: 'charmap' codec can't encode character '\u2191' in position 195: character maps to <undefined>
**Solution:**
```
set PYTHONIOENCODING=UTF-8 # for windows
export PYTHONIOENCODING=UTF-8 # for linux
```

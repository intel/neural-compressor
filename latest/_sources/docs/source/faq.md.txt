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

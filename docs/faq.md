Frequently Asked Questions
===
## Common Build Issues
#### Issue 1:  
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject     
**Solution:** reinstall pycocotools by "pip install pycocotools --no-cache-dir"  
#### Issue 2:  
ImportError: libGL.so.1: cannot open shared object file: No such file or directory   
**Solution:** apt install or yum install opencv
#### Issue 3:  
Conda package *neural-compressor-full* dependency conflict will pending conda installation for a long time.
**Solution:** run *conda install sqlalchemy=1.4.27 alembic=1.7.7 -c conda-forge* before install *neural-compressor-full*. 

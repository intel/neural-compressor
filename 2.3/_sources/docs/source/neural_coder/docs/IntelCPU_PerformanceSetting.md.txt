## Intel CPU Platforms: Best Performance Setting
### Install MKL, OpenMP and JEMALLOC
The simplest way for installation is through ```conda install```:
```bash
conda install -y mkl mkl-include jemalloc
```

### Install NUMA Controller
```bash
apt-get update && apt-get install bc numactl
```

### Environment Variables
Check if your ```CONDA_PREFIX``` has a value by:
```bash
echo ${CONDA_PREFIX}
```
If it is empty, it means that you are not in a traditional CONDA environment, you need to find the location of the ```.so``` files by:
```bash
find / -name "libjemalloc.so"
find / -name "libiomp5.so"
```
It will show the path these files were installed into. For example:
```bash
/home/name/lib/libjemalloc.so
/home/name/lib/libiomp5.so
```
And then you should ```export``` this path as ```CONDA_PREFIX```:
```bash
export CONDA_PREFIX="/home/name"
```
Finally:
```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
```

### Frequency Governers
Check the frequency governor state on your machine:
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
If it shows ```powersave``` instead of ```performance```, execute:
```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

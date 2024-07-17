Benchmark
---

1. [Introduction](#introduction)

2. [Supported Matrix](#supported-matrix)

3. [Usage](#usage)

## Introduction

Intel Neural Compressor provides a command `incbench` to launch the Intel CPU performance benchmark.

To get the peak performance on Intel Xeon CPU, we should avoid crossing NUMA node in one instance.
Therefore, by default, `incbench` will trigger 1 instance on the first NUMA node.

## Supported Matrix

| Platform | Status |
|:---:|:---:|
| Linux   | &#10004; |
| Windows | &#10004; |

## Usage

|       Parameters       |          Default         |                comments               |
|:----------------------:|:------------------------:|:-------------------------------------:|
|      num_instances     |             1            |          Number of instances          |
| num_cores_per_instance |           None           |    Number of cores in each instance   |
|        C, cores        | 0-${num_cores_on_NUMA-1} |     decides the visible core range    |
|      cross_memory      |           False          | whether to allocate memory cross NUMA |

> Note: cross_memory is set to True only when memory is insufficient.

### General Use Cases

1. `incbench main.py`: run 1 instance on NUMA:0.
2. `incbench --num_i 2 main.py`: run 2 instances on NUMA:0.
3. `incbench --num_c 2 main.py`: run multi-instances with 2 cores per instance on NUMA:0.
4. `incbench -C 24-47 main.py`: run 1 instance on COREs:24-47.
5. `incbench -C 24-47 --num_c 4 main.py`: run multi-instances with 4 COREs per instance on COREs:24-47.

> Note:
    > - `num_i` works the same as `num_instances`
    > - `num_c` works the same as `num_cores_per_instance`

### Dump Throughput and Latency Summary

To merge benchmark results from multi-instances, "incbench" automatically checks log file messages for "throughput" and "latency" information matching the following patterns.

```python
throughput_pattern = r"[T,t]hroughput:\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z/]*)"
latency_pattern = r"[L,l]atency:\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z/]*)"
```

#### Demo usage

```python
print("Throughput: {:.3f} samples/sec".format(throughput))
print("Latency: {:.3f} ms".format(latency * 10**3))
```

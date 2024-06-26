Benchmark
---

1. [Introduction](#introduction)

2. [Supported Parameters](#supported-parameters)

3. [General Use Cases](#general-use-cases)

## Introduction

Intel Neural Compressor provides a command `incbench` to launch the Intel CPU performance benchmark.

To get the peak performance on Intel Xeon CPU, we should avoid crossing NUMA node in one instance.
Therefore, by default, `incbench` will trigger 1 instance on the first NUMA node.

## Supported Parameters

|       Parameters       |          Default         |                comments               |
|:----------------------:|:------------------------:|:-------------------------------------:|
|      num_instances     |             1            |          Number of instances          |
| num_cores_per_instance |           None           |    Number of cores in each instance   |
|        C, cores        | 0-${num_cores_on_NUMA-1} |     decides the visible core range    |
|      cross_memory      |           False          | whether to allocate memory cross NUMA |

> Note: cross_memory is set to True only when memory is insufficient.

## General Use Cases

1. `incbench main.py`: run 1 instance on NUMA:0.
2. `incbench --num_i 2 main.py`: run 2 instances on NUMA:0.
3. `incbench --num_c 2 main.py`: run multi-instances with 2 cores per instance on NUMA:0.
4. `incbench -C 24-47 main.py`: run 1 instance on COREs:24-47.
5. `incbench -C 24-47 --num_c 4 main.py`: run multi-instances with 4 COREs per instance on COREs:24-47.

> Note:
    > - `num_i` works the same as `num_instances`
    > - `num_c` works the same as `num_cores_per_instance`

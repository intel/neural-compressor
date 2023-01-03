Benchmarking
============
1. [Introduction](#Introduction)
2. [Benchmark Support Matrix](#Benchmark-Support-Matrix)
3. [Get Started with Benchmark](#Get-Started-with-Benchmark)
4. [Examples](#Examples)

## Introduction
The benchmarking feature of Neural Compressor is used to measure the model performance with the objective settings. 
Users can get the performance of the float32 model and the optimized low precision model in the same scenarios.

## Benchmark Support Matrix
<table>
    <thead>
        <tr>
            <th>Environment</th>
            <th>Category</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Operating System</td>
            <td> linux </td>
        </tr>
        <tr>
            <td> windows </td>
        </tr>
        <tr>
            <td rowspan=3> Architecture </td>
            <td> x86_64 </td>
        </tr>
        <tr>
            <td> aarch64 </td>
        </tr>
        <tr>
            <td> gpu </td>
        </tr>
    </tbody>
</table>

## Get Started with Benchmark API

Benchmark provide capability to automatically run with multiple instance through `cores_per_instance` and `num_of_instance` config (CPU only). 
And please make sure `cores_per_instance * num_of_instance` must be less than CPU physical core numbers. 
`benchmark.fit` accept `b_dataloader` or `b_func` as input. 
`b_func` is customized benchmark function. If user passes the `b_dataloader`, then `b_func` is not required.

```python
from neural_compressor.config import BenchmarkConfig
from neural_compressor.benchmark import fit
conf = BenchmarkConfig(warmup=10, iteration=100, cores_per_instance=4, num_of_instance=7)
fit(model='./int8.pb', config=conf, b_dataloader=eval_dataloader)
```

## Examples

Refer to the [Benchmark example](../../examples/helloworld/tf_example5).


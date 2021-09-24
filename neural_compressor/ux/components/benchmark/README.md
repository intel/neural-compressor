# Execute benchmark endpoint

## Request
**Path**: `/api/benchmark`

**Body parameters**:

| Parameter | Description |
|:-----------|:-------------|
| **id** | workload ID |
| **workspace_path** | path to current UX workspace |
| **input_model** | input model to benchmark |
| **optimized_model** | optimized model to benchmark |

> *Model definition*:
> ```json
> {
>     "precision": "fp32",  // Model datatype
>     "path": "/path/to/fp32_model.pb"  // Model path
>     "mods": ["benchmark", "accuracy"] // Benchmark mode
> }
> ```

## Example

**Body**: 
 ```json
{
    "id": "configuration_id",
    "workspace_path": "/path/to/workspace",
    "input_model": {
        "precision": "fp32",
        "path": "/localdisk/fp32.pb"
    },
    "optimized_model": {
        "precision": "int8",
        "path": "/localdisk/int8.pb"
    }
}
```


**Responses**
1) **Subject**: `benchmark_start`

    **Data**:
    ```json
    {
        "id": "configuration_id",
        "message": "started"
    }
    ```
2) **Subject**: `benchmark_progress`

    **Data**:
    ```json
    {
        "id": "configuration_id",
        "execution_details": {
            "input_model_benchmark": {
                "performance": {
                    "instances": 7,
                    "cores_per_instance": 4,
                    "model_path": "/localdisk/fp32.pb",
                    "precision": "fp32",
                    "mode": "performance",
                    "batch_size": 1,
                    "framework": "tensorflow",
                    "config_path": "/path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml",
                    "benchmark_script": "/localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py",
                    "command": "python /localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py --config /path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml --input-graph /localdisk/fp32.pb --mode performance --framework tensorflow"
                }
            }
        },
        "progress": "1/2",
        "perf_throughput_input_model": <float>
    }
    ```

3) **Subject**: `benchmark_progress`

    **Data**:
    ```json
    {
        "id": "configuration_id",
        "execution_details": {
            "input_model_benchmark": {
                "performance": {
                    "instances": 7,
                    "cores_per_instance": 4,
                    "model_path": "/localdisk/fp32.pb",
                    "precision": "fp32",
                    "mode": "performance",
                    "batch_size": 1,
                    "framework": "tensorflow",
                    "config_path": "/path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml",
                    "benchmark_script": "/localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py",
                    "command": "python /localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py --config /path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml --input-graph /localdisk/fp32.pb --mode performance --framework tensorflow"
                }
            },
            "optimized_model_benchmark": {
                "performance": {
                    "instances": 7,
                    "cores_per_instance": 4,
                    "model_path": "/localdisk/int8.pb",
                    "precision": "int8",
                    "mode": "performance",
                    "batch_size": 1,
                    "framework": "tensorflow",
                    "config_path": "/path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml",
                    "benchmark_script": "/localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py",
                    "command": "python /localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py --config /path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml --input-graph /localdisk/int8.pb --mode performance --framework tensorflow"
                }
            }
        },
        "progress": "2/2",
        "perf_throughput_input_model": <float>,
        "perf_throughput_optimized_model": <float>
    }
    ```

4) **Subject**: `benchmark_finish`

    **Data**:
    ```json
    {
        "id": "configuration_id",
        "execution_details": {
            "input_model_benchmark": {
                "performance": {
                    "instances": 7,
                    "cores_per_instance": 4,
                    "model_path": "/localdisk/fp32.pb",
                    "precision": "fp32",
                    "mode": "performance",
                    "batch_size": 1,
                    "framework": "tensorflow",
                    "config_path": "/path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml",
                    "benchmark_script": "/localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py",
                    "command": "python /localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py --config /path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml --input-graph /localdisk/fp32.pb --mode performance --framework tensorflow"
                }
            },
            "optimized_model_benchmark": {
                "performance": {
                    "instances": 7,
                    "cores_per_instance": 4,
                    "model_path": "/localdisk/int8.pb",
                    "precision": "int8",
                    "mode": "performance",
                    "batch_size": 1,
                    "framework": "tensorflow",
                    "config_path": "/path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml",
                    "benchmark_script": "/localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py",
                    "command": "python /localdisk/neural_compressor/ux/components/benchmark/benchmark_model.py --config /path/to/workspace/workloads/<model_name>_<configuration_id>/config.yaml --input-graph /localdisk/int8.pb --mode performance --framework tensorflow"
                }
            }
        },
        "progress": "2/2",
        "perf_throughput_input_model": <float>,
        "perf_throughput_optimized_model": <float>
    }
    ```

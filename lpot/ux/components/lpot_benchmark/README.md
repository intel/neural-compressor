# Execute benchmark endpoint

## Request
**Path**: `/api/benchmark`

**Body parameters**:

| Parameter | Description |
|:-----------|:-------------|
| **id** | workload ID |
| **workspace_path** | path to LPOT UX current workspace |
| **models** | list of models to benchmark |

> *Model definition*:
> ```json
> {
>     "precision": "fp32",  // Model datatype
>     "path": "/path/to/fp32_model.pb"  // Model path
> }
> ```

## Example

**Body**: 
 ```json
{
    "id": "configuration_id",
    "workspace_path": "/path/to/workspace",
    "models": [
        {
            "precision": "fp32",
            "path": "/path/to/fp32_model.pb"
        },
        {
            "precision": "int8",
            "path": "/path/to/int8_model.pb"
        }
    ]
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
        "id": "22b60ef39915ff4931936ca43b8c3ead",
        "perf_throughput_fp32": 85.672,
        "progress": "1/2"
    }
    ```

3) **Subject**: `benchmark_progress`

    **Data**:
    ```json
    {
        "id": "22b60ef39915ff4931936ca43b8c3ead",
        "perf_throughput_fp32": 85.672,
        "progress": "2/2",
        "perf_throughput_int8": 241.698
    }
    ```

4) **Subject**: `benchmark_finish`

    **Data**:
    ```json
    {
        "id": "22b60ef39915ff4931936ca43b8c3ead",
        "perf_throughput_fp32": 85.672,
        "progress": "2/2",
        "perf_throughput_int8": 241.698
    }
    ```
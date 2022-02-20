# Intel® Neural Compressor Bench 

## Table of Contents
- [Endpoints](#endpoints)
  - [Project](#project)
    - [Create project](#create-project)
    - [Get project details](#get-project-details)
    - [Add notes to project](#add-notes-to-project)
    - [List projects](#list-projects)
  - [Dataset](#dataset)
    - [Add dataset to project](#add-dataset-to-project)
    - [Get dataset details](#get-dataset-details)
    - [List datasets](#list-datasets)
  - [Optimization](#optimization)
    - [Add optimization to project](#add-optimization-to-project)
    - [Get optimization details](#get-optimization-details)
    - [List optimizations](#list-optimizations)
    - [Execute optimization](#execute-optimization)
    - [Pin accuracy benchmark to optimization](#pin-accuracy-benchmark-to-optimization)
    - [Pin performance benchmark to optimization](#pin-performance-benchmark-to-optimization)
  - [Benchmark](#benchmark)
    - [Add benchmark to project](#add-benchmark-to-project)
    - [Get benchmark details](#get-benchmark-details)
    - [List benchmarks](#list-benchmarks)
    - [Execute benchmark](#execute-benchmark)
  - [Profiling](#profiling)
    - [Add profiling to project](#add-profiling-to-project)
    - [Get profiling details](#get-profiling-details)
    - [List profilings](#list-profilings)
    - [Execute profiling](#execute-profiling)
  - [Model](#model)
    - [List models](#list-models)
    - [Get model's boundary nodes](#get-models-boundary-nodes)
    - [Get model's graph](#get-models-graph)
  - [Dictionaries](#dictionaries)
    - [Domains](#domains)
    - [Domain Flavours](#domain-flavours)
    - [Optimization Types](#optimization-types)
    - [Optimization Types and support for specific precision](#optimization-types-and-support-for-specific-precision)
    - [Precisions](#precisions)
    - [Dataloaders](#dataloaders)
    - [Transforms](#transforms)
  
## Endpoints

### Project
#### Create project
**Path**: `api/project/create`<br>
**Method**: `POST`<br>
**Returns**: new project id and model id

**Parameters description:**
<table>
    <thead>
        <tr>
            <th colspan=2>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=2><b>name</b></td>
            <td>Project name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td rowspan=5><b>model</b></td>
            <td><b>path</b></td>
            <td>Model path</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>input_nodes</b></td>
            <td>Comma separated list of model's input nodes (without brackets)</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>output_nodes</b></td>
            <td>Comma separated list of model's output nodes (without brackets)</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>domain</b></td>
            <td>String with domain name.<br>Supported domains: <code>Image Recognition</code>, <code>Object Detection</code>, <code>Neural Language Processing</code>, <code>Recommendation</code></td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>shape</b></td>
            <td>Comma separated input shapes</td>
            <td align="center">&#x2610;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "name": "My new project",
    "model": {
        "path": "/path/to/model.pb",
        "input_nodes": "input1,input2",
        "output_nodes": "output",
        "domain": "Image Recognition"
    }
}
```


**Example response**:
```json
{
    "project_id": 1,
    "model_id": 1
}
```

#### Get project details
**Path**: `api/project`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1
}
```

**Example response**:
```json
{
    "id": 1,
    "name": "My new project",
    "notes": null,
    "created_at": "Tue, 04 Jan 2022 16:09:29 GMT",
    "modified_at": null,
    "input_model": {
        "id": 1,
        "name": "Input model",
        "path": "/path/to/model.pb",
        "framework": {
            "id": 1,
            "name": "TensorFlow"
        },
        "size": 98.0,
        "precision": {
            "id": 2,
            "name": "fp32"
        },
        "domain": {
            "id": 1,
            "name": "Image Recognition"
        },
        "domain_flavour": {
            "id": 1,
            "name": ""
        },
        "input_nodes": [
            "input_tensor"
        ],
        "output_nodes": [
            "softmax_tensor"
        ],
        "supports_graph": true,
        "supports_profiling": true,
        "created_at": "Tue, 04 Jan 2022 16:09:30 GMT",
    }
}
```
#### Add notes to project
**Path**: `api/project/note`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>notes</b></td>
            <td>Project notes</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
    "notes": "Project notes"
}
```

**Example response**:
```json
{
    "id": 1,
    "notes": "Project notes"
}
```
#### List projects
**Path**: `api/project/list`<br>
**Method**: `GET` 

**Example response**:
```json
{
    "projects": [
        {
            "id": 1,
            "name": "My new project",
            "created_at": "Fri, 10 Dec 2021 13:19:26 GMT",
            "modified_at": null
        }
    ]
}
```

### Dataset
#### Add dataset to project
**Path**: `api/dataset/add`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>Dataset name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>dataset_path</b></td>
            <td>Path to the dataset</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>transform</b></td>
            <td>List of transformations</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>metric</b></td>
            <td>Metric name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>metric_param</b></td>
            <td>Metric parameter</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1,
    "name": "My Dataset",
    "dataset_path": "/path/to/dataset",
    "transform": [
        {
            "name": "ResizeCropImagenet",
            "params": {
                "height": 224,
                "width": 224,
                "mean_value": [
                    123.68,
                    116.78,
                    103.94
                ]
            }
        }
    ],
    "dataloader": {
        "name": "ImageRecord",
        "params": {
            "root": ""
        }
    },
    "metric": "topk",
    "metric_param": 1
}
```

**Example response**:
```json
{
    "dataset_id": 2
}
```

#### Get dataset details
**Path**: `api/dataset`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Dataset id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1
}
```

**Example response**:
```json
{
    "id": 1,
    "project_id": 1,
    "name": "dummy",
    "dataset_type": "dummy_v2",
    "parameters": {
        "input_shape": [
            [
                224,
                224,
                3
            ]
        ],
        "label_shape": [
            1
        ]
    },
    "transforms": {},
    "metric": {},
    "calibration_batch_size": 100,
    "calibration_sampling_size": 100,
    "created_at": "Thu, 16 Dec 2021 08:41:33 GMT"
}
```

#### List datasets
**Path**: `api/dataset/list`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1
}
```

**Example response**:
```json
{
    "datasets": [
        {
            "id": 1,
            "name": "dummy",
            "dataset_type": "dummy_v2",
            "created_at": "Thu, 16 Dec 2021 08:41:33 GMT"
        }
    ]
}
```
 
### Optimization
#### Add optimization to project
**Path**: `api/optimization/add`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>Optimization name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>precision_id</b></td>
            <td>ID of precision</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>optimization_type_id</b></td>
            <td>ID of optimization type</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>dataset_id</b></td>
            <td>ID of dataset</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1,
    "name": "My optimization",
    "precision_id": 1,
    "optimization_type_id": 1,
    "dataset_id": 1
}
```

**Example response**:
```json
{
    "optimization_id": 1
}
```

#### Get optimization details
**Path**: `api/optimization`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Optimization id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
}
```

**Example response**:
```json
{
    "project_id": 1,
    "id": 1,
    "name": "My optimization",
    "precision": {
        "id": 1,
        "name": "int8"
    },
    "optimization_type": {
        "id": 1,
        "name": "Quantization"
    },
    "dataset": {
        "id": 1,
        "name": "dummy"
    },
    "config_path": "/path/to/config.yaml",
    "log_path": "/path/to/output.txt",
    "execution_command": "python tune_model.py --input-graph /path/to/model.pb --output-graph /home/workdir/models/my_optimization_1/my_optimization.pb --config /path/to/config.yaml --framework tensorflow",
    "batch_size": 100,
    "sampling_size": 100,
    "status": "success",
    "created_at": "2022-01-28 12:37:01",
    "last_run_at": "2022-01-28 12:37:32",
    "duration": 16,
    "optimized_model": {
        "id": 2,
        "name": "my_optimization",
        "path": "/home/workdir/models/my_optimization_1/my_optimization.pb",
        "framework": {
            "id": 1,
            "name": "TensorFlow"
        },
        "size": 25.0,
        "precision": {
            "id": 1,
            "name": "int8"
        },
        "domain": {
            "id": 1,
            "name": "Image Recognition"
        },
        "domain_flavour": {
            "id": 1,
            "name": ""
        },
        "input_nodes": [
            "input_tensor"
        ],
        "output_nodes": [
            "softmax_tensor"
        ],
        "supports_graph": true,
        "supports_profiling": true,
        "created_at": "2022-01-28 12:37:32"
    },
    "accuracy_benchmark": null,
    "performance_benchmark": null,
    "tuning_details": {
        "strategy": "basic",
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_threshold": 0.1,
        "multi_objective": "performance",
        "exit_policy": {
            "timeout": 0
        },
        "random_seed": 9527,
        "created_at": "2022-01-28 12:37:01",
        "modified_at": null
    }
}
```

#### List optimizations
**Path**: `api/optimization/list`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1,
}
```

**Example response**:
```json
{
    "optimizations": [
        {
            "project_id": 1,
            "id": 1,
            "name": "My graph optimization",
            "precision": {
                "id": 2,
                "name": "fp32"
            },
            "optimization_type": {
                "id": 2,
                "name": "Graph optimization"
            },
            "dataset": {
                "id": 2,
                "name": "COCORecord_50"
            },
            "config_path": null,
            "log_path": "/path/to/output.txt",
            "execution_command": "python optimize_model.py --input-graph=/path/to/model.pb --output-graph=/home/workdir/models/my_graph_optimization_1/my_graph_optimization.pb --framework=tensorflow --input-nodes=['image_tensor'] --output-nodes=['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'] --precisions=fp32",
            "batch_size": 100,
            "sampling_size": 100,
            "status": "success",
            "created_at": "2022-01-28 12:22:19",
            "last_run_at": "2022-01-28 12:23:16",
            "duration": 12,
            "optimized_model": {
                "id": 3,
                "name": "my_graph_optimization",
                "path": "/home/workdir/models/my_graph_optimization_1/my_graph_optimization.pb",
                "framework": {
                    "id": 1,
                    "name": "TensorFlow"
                },
                "size": 27.0,
                "precision": {
                    "id": 2,
                    "name": "fp32"
                },
                "domain": {
                    "id": 2,
                    "name": "Object Detection"
                },
                "domain_flavour": {
                    "id": 1,
                    "name": ""
                },
                "input_nodes": [
                    "image_tensor"
                ],
                "output_nodes": [
                    "num_detections",
                    "detection_boxes",
                    "detection_scores",
                    "detection_classes"
                ],
                "supports_graph": true,
                "supports_profiling": true,
                "created_at": "2022-01-28 12:23:16"
            },
            "accuracy_benchmark": null,
            "performance_benchmark": null,
            "tuning_details": null
        },
        {
            "project_id": 1,
            "id": 2,
            "name": "My optimization",
            "precision": {
                "id": 1,
                "name": "int8"
            },
            "optimization_type": {
                "id": 1,
                "name": "Quantization"
            },
            "dataset": {
                "id": 2,
                "name": "dummy"
            },
            "config_path": "/path/to/config.yaml",
            "log_path": "/path/to/output.txt",
            "execution_command": "python tune_model.py --input-graph /path/to/model.pb --output-graph /home/workdir/models/my_optimization_2/my_optimization.pb --config /path/to/config.yaml --framework tensorflow",
            "batch_size": 100,
            "sampling_size": 100,
            "status": "success",
            "created_at": "2022-01-28 12:37:01",
            "last_run_at": "2022-01-28 12:37:32",
            "duration": 16,
            "optimized_model": {
                "id": 4,
                "name": "my_first_optimization",
                "path": "/home/workdir/models/my_first_optimization_2/my_first_optimization.pb",
                "framework": {
                    "id": 1,
                    "name": "TensorFlow"
                },
                "size": 25.0,
                "precision": {
                    "id": 1,
                    "name": "int8"
                },
                "domain": {
                    "id": 2,
                    "name": "Object Detection"
                },
                "domain_flavour": {
                    "id": 1,
                    "name": ""
                },
                "input_nodes": [
                    "image_tensor"
                ],
                "output_nodes": [
                    "num_detections",
                    "detection_boxes",
                    "detection_scores",
                    "detection_classes"
                ],
                "supports_graph": true,
                "supports_profiling": true,
                "created_at": "2022-01-28 12:37:32"
            },
            "accuracy_benchmark": null,
            "performance_benchmark": null,
            "tuning_details": null
        }
    ]
}
```

#### Execute optimization
**Path**: `api/optimization/execute`<br>
**Method**: `POST`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>optimization_id</b></td>
            <td>ID of optimization</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "asd",
    "optimization_id": 1
}
```

**Example response**:
```json
{
    "exit_code": 102,
    "message": "processing"
}
```
**Responses over WebSockets**:
```json
[
    "optimization_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "request_id": "asd",
            "size_input_model": 98,
            "config_path": "/path/to/config.yaml",
            "output_path": "/path/to/output.txt"
        }
    }
]
```
```json
[
    "optimization_finish",
    {
        "status": "success",
        "data": {
            "request_id": "asd",
            "project_id": 1,
            "id": 1,
            "name": "My optimization",
            "precision": {
                "id": 1,
                "name": "int8"
            },
            "optimization_type": {
                "id": 1,
                "name": "Quantization"
            },
            "dataset": {
                "id": 1,
                "name": "dummy"
            },
            "config_path": "/path/to/config.yaml",
            "log_path": "/path/to/output.txt",
            "execution_command": "python tune_model.py --input-graph /path/to/model.pb --output-graph /home/workdir/models/my_optimization_1/my_optimization.pb --config /path/to/config.yaml --framework tensorflow",
            "batch_size": 100,
            "sampling_size": 100,
            "status": "success",
            "created_at": "2022-01-28 12:37:01",
            "last_run_at": "2022-01-28 12:37:32",
            "duration": 16,
            "optimized_model": {
                "id": 2,
                "name": "my_optimization",
                "path": "/home/workdir/models/my_optimization_1/my_optimization.pb",
                "framework": {
                    "id": 1,
                    "name": "TensorFlow"
                },
                "size": 25.0,
                "precision": {
                    "id": 1,
                    "name": "int8"
                },
                "domain": {
                    "id": 1,
                    "name": "Image Recognition"
                },
                "domain_flavour": {
                    "id": 1,
                    "name": ""
                },
                "input_nodes": [
                    "input_tensor"
                ],
                "output_nodes": [
                    "softmax_tensor"
                ],
                "supports_graph": true,
                "supports_profiling": true,
                "created_at": "2022-01-28 12:37:32"
            },
            "accuracy_benchmark": null,
            "performance_benchmark": null,
            "tuning_details": {
                "strategy": "basic",
                "accuracy_criterion_type": "relative",
                "accuracy_criterion_threshold": 0.1,
                "multi_objective": "performance",
                "exit_policy": {
                    "timeout": 0
                },
                "random_seed": 9527,
                "created_at": "2022-01-28 12:37:01",
                "modified_at": null
            }
        }
    }
]
```



#### Pin accuracy benchmark to optimization
**Path**: `api/optimization/pin_accuracy_benchmark`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>optimization_id</b></td>
            <td>Optimization id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>benchmark_id</b></td>
            <td>Benchmark id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "optimization_id": 1,
    "benchmark_id": 2
}
```

**Example response**:
```json
{
    "id": 1,
    "accuracy_benchmark_id": 2
}
```

#### Pin performance benchmark to optimization
**Path**: `api/optimization/pin_performance_benchmark`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>optimization_id</b></td>
            <td>Optimization id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>benchmark_id</b></td>
            <td>Benchmark id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "optimization_id": 1,
    "benchmark_id": 3
}
```

**Example response**:
```json
{
    "id": 1,
    "performance_benchmark_id": 3
}
```

### Benchmark
#### Add benchmark to project
**Path**: `api/benchmark/add`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>Benchmark name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>model_id</b></td>
            <td>ID of model to benchmark</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>dataset_id</b></td>
            <td>ID of dataset</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>batch_size</b></td>
            <td>Benchmark batch size</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>warmup_iterations</b></td>
            <td>Benchmark warmup iterations</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>iterations</b></td>
            <td>Benchmark iterations</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>number_of_instance</b></td>
            <td>Number of benchmark instances</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>cores_per_instance</b></td>
            <td>Number of cores per benchmark instances</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "name": "My benchmark",
    "project_id": 1,
    "model_id": 1,
    "dataset_id": 1,
    "batch_size": 32,
    "iterations": -1,
    "number_of_instance": 1,
    "cores_per_instance": 4,
    "warmup_iterations": 10
}
```

**Example response**:
```json
{
    "benchmark_id": 1
}
```

#### Get benchmark details
**Path**: `api/benchmark`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Benchmark id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1
}
```

**Example response**:
```json
{
    "project_id": 1,
    "id": 1,
    "name": "Benchmark of input model",
    "model": {
        "id": 1,
        "name": "Input model",
        "path": "/path/to/model.pb",
        "framework": {
            "id": 1,
            "name": "TensorFlow"
        },
        "size": 98.0,
        "precision": {
            "id": 2,
            "name": "fp32"
        },
        "domain": {
            "id": 1,
            "name": "Image Recognition"
        },
        "domain_flavour": {
            "id": 1,
            "name": ""
        },
        "input_nodes": "input_tensor",
        "output_nodes": "softmax_tensor",
        "supports_graph": true,
        "supports_profiling": true,
        "created_at": "2022-01-31 08:57:36"
    },
    "dataset": {
        "id": 1,
        "project_id": 1,
        "name": "dummy",
        "dataset_type": "dummy_v2",
        "parameters": {
            "input_shape": [
                [
                    224,
                    224,
                    3
                ]
            ],
            "label_shape": [
                1
            ]
        },
        "transforms": null,
        "filter": null,
        "metric": null,
        "template_path": null,
        "created_at": "2022-01-31 08:57:39"
    },
    "result": null,
    "batch_size": 32,
    "warmup_iterations": 10,
    "iterations": -1,
    "number_of_instance": 1,
    "cores_per_instance": 4,
    "created_at": "Mon, 31 Jan 2022 08:57:46 GMT",
    "last_run_at": null,
    "duration": null
}
```

#### List benchmarks
**Path**: `api/benchmark/list`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1
}
```

**Example response**:
```json
{
    "benchmarks": [
        {
            "project_id": 1,
            "id": 1,
            "name": "Benchmark of input model",
            "model": {
                ...
            },
            "dataset": {
                ...
            },
            "result": null,
            "batch_size": 32,
            "warmup_iterations": 10,
            "iterations": -1,
            "number_of_instance": 1,
            "cores_per_instance": 4,
            "created_at": "Mon, 31 Jan 2022 08:44:37 GMT",
            "last_run_at": null,
            "duration": null
        },
        {
            "project_id": 1,
            "id": 2,
            "name": "Optimized model benchmark",
            "model": {
                ...
            },
            "dataset": {
                ...
            },
            "result": null,
            "batch_size": 16,
            "warmup_iterations": 10,
            "iterations": -1,
            "number_of_instance": 1,
            "cores_per_instance": 4,
            "created_at": "Mon, 31 Jan 2022 08:44:37 GMT",
            "last_run_at": null,
            "duration": null
        }
    ]
}
```

#### Execute benchmark
**Path**: `api/benchmark/execute`<br>
**Method**: `POST`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>benchmark_id</b></td>
            <td>ID of benchmark</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "asd",
    "benchmark_id": 1
}
```

**Example response**:
```json
{
    "exit_code": 102,
    "message": "processing"
}
```
**Responses over WebSockets**:
```json
[
    "benchmark_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "request_id": "asd",
            "config_path": "/path/to/config.yaml",
            "output_path": "/path/to/output.txt"
        }
    }
]
```
```json
[
    "benchmark_finish",
    {
        "status": "success",
        "data": {
            "request_id": "asd",
            "project_id": 1,
            "id": 1,
            "name": "Benchmark of optimized model",
            "model": {
                "id": 1,
                "name": "Input model",
                "path": "/path/to/model.pb",
                "framework": {
                    "id": 1,
                    "name": "TensorFlow"
                },
                "size": 98,
                "precision": {
                    "id": 2,
                    "name": "fp32"
                },
                "domain": {
                    "id": 1,
                    "name": "Image Recognition"
                },
                "domain_flavour": {
                    "id": 1,
                    "name": ""
                },
                "input_nodes": "input_tensor",
                "output_nodes": "softmax_tensor",
                "supports_graph": true,
                "supports_profiling": true,
                "created_at": "2022-01-31 15:37:16"
            },
            "dataset": {
                "id": 1,
                "project_id": 1,
                "name": "dummy",
                "dataset_type": "dummy_v2",
                "parameters": {
                    "input_shape": [
                        [
                            224,
                            224,
                            3
                        ]
                    ],
                    "label_shape": [
                        1
                    ]
                },
                "transforms": null,
                "filter": null,
                "metric": null,
                "template_path": null,
                "created_at": "2022-01-31 15:37:19"
            },
            "mode": "performance",
            "result": {
                "benchmark_id": 1,
                "id": 1,
                "accuracy": null,
                "performance": 46.412,
                "created_at": "2022-01-31 15:38:14",
                "last_run_at": "null",
                "duration": null
            },
            "batch_size": 1,
            "warmup_iterations": 5,
            "iterations": -1,
            "number_of_instance": 1,
            "cores_per_instance": 4,
            "config_path": "/path/to/config.yaml",
            "log_path": "/pat/to/output.txt",
            "execution_command": "python benchmark_model.py --config /path/to/config.yaml --input-graph /path/to/model.pb --mode performance --framework tensorflow",
            "status": "success",
            "created_at": "2022-01-31 15:38:01",
            "last_run_at": "2022-01-31 16:06:04",
            "duration": 9
        }
    }
]
```

### Profiling
#### Add profiling to project
**Path**: `api/profiling/add`<br>
**Method**: `POST`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>Profiling name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>model_id</b></td>
            <td>ID of model to profile</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>dataset_id</b></td>
            <td>ID of dataset</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>num_threads</b></td>
            <td>Number of threads to use in profiling</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1,
    "name": "Input model profiling",
    "model_id": 1,
    "dataset_id": 1,
    "num_threads": 7
}
```

**Example response**:
```json
{
    "profiling_id": 1
}
```

#### Get profiling details
**Path**: `api/profiling`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>id</b></td>
            <td>Profiling id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1
}
```

**Example response**:
```json
{
    "project_id": 1,
    "id": 1,
    "name": "Input model profiling",
    "model": {
        .
        .
        .
    },
    "dataset": {
        .
        .
        .
    },
    "num_threads": 7,
    "execution_command": null,
    "log_path": null,
    "status": null,
    "created_at": "2022-02-07 14:24:23",
    "last_run_at": "None",
    "duration": null
}
```

#### List profilings
**Path**: `api/profiling/list`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1
}
```

```json
{
    "profilings": [
        {
            "project_id": 1,
            "id": 1,
            "name": "Input model profiling",
            "model": {
               .
               .
               .
            },
            "dataset": {
               .
               .
               .
            },
            "num_threads": 7,
            "execution_command": "python /path/to/profile_model.py --profiling-config /path/to/config.json",
            "log_path": "/path/to/output.txt",
            "status": "error",
            "created_at": "2022-02-07 13:35:00",
            "last_run_at": "2022-02-07 13:38:35",
            "duration": 64
        },
        {
            "project_id": 1,
            "id": 2,
            "name": "Profiling of optimized model",
            "model": {
               .
               .
               .
            },
            "dataset": {
                .
                .
                .
            },
            "num_threads": 7,
            "execution_command": null,
            "log_path": null,
            "status": null,
            "created_at": "2022-02-07 14:24:23",
            "last_run_at": "None",
            "duration": null
        }
    ]
}
```

#### Execute profiling
**Path**: `api/profiling/execute`<br>
**Method**: `POST`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>profiling_id</b></td>
            <td>ID of profiling</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "asd",
    "profiling_id": 1
}
```

**Example response**:
```json
{
    "exit_code": 102,
    "message": "processing"
}
```
**Responses over WebSockets**:
```json
[
    "profiling_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "request_id": "asd",
            "output_path": "/path/to/output.txt"
        }
    }
]
```
```json
[
    "profiling_finish",
    {
        "status": "success",
        "data": {
            "request_id": "asd",
            "project_id": 1,
            "id": 1,
            "name": "Input model profiling",
            "model": {
                .
                .
                .
            },
            "dataset": {
                .
                .
                .
            },
            "num_threads": 7,
            "execution_command": "python /path/to/profile_model.py --profiling-config /path/to/config.json",
            "log_path": "/path/to/output.txt",
            "status": "success",
            "created_at": "2022-02-07 14:24:23",
            "last_run_at": "2022-02-07 14:42:28",
            "duration": 68,
            "results": [
                {
                    "profiling_id": 1,
                    "id": 54,
                    "node_name": "Add",
                    "total_execution_time": <time in micro seconds>,
                    "accelerator_execution_time": <time in micro seconds>,
                    "cpu_execution_time": <time in micro seconds>,
                    "op_run": <num of op occurrences>,
                    "op_defined": <num of op occurrences>,
                    "created_at": "2022-02-07 14:42:28"
                },
                {
                    "profiling_id": 1,
                    "id": 53,
                    "node_name": "BiasAdd",
                    "total_execution_time": <time in micro seconds>,
                    "accelerator_execution_time": <time in micro seconds>,
                    "cpu_execution_time": <time in micro seconds>,
                    "op_run": <num of op occurrences>,
                    "op_defined": <num of op occurrences>,
                    "created_at": "2022-02-07 14:42:28"
                },
                .
                .
                .
                {
                    "profiling_id": 1,
                    "id": 63,
                    "node_name": "Squeeze",
                    "total_execution_time": <time in micro seconds>,
                    "accelerator_execution_time": <time in micro seconds>,
                    "cpu_execution_time": <time in micro seconds>,
                    "op_run": <num of op occurrences>,
                    "op_defined": <num of op occurrences>,
                    "created_at": "2022-02-07 14:42:28"
                }
            ]
        }
    }
]
```

### Model
#### List models
**Path**: `api/model/list`<br>
**Method**: `GET`

**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>project_id</b></td>
            <td>Project id</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "project_id": 1
}
```

**Example response**:
```json
{
    "models": [
        {
            "id": 1,
            "name": "Input model",
            "path": "/path/to/model.pb",
            "precision_id": 2,
            "created_at": "Thu, 16 Dec 2021 08:41:28 GMT"
        }
    ]
}
```

#### Get model's boundary nodes
**Path**: `api/model/boundary_nodes`<br>
**Method**: `POST`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>model_path</b></td>
            <td>Path to model</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "asd",
    "model_path": "/path/to/model.pb"
}
```

**Example response**:
```json
{
    "exit_code": 102,
    "message": "processing"
}
```
**Responses over WebSockets**:
```json
[
    "boundary_nodes_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "id": "asd"
        }
    }
]
```
```json
[
    "boundary_nodes_finish",
    {
        "status": "success",
        "data": {
            "id": "asd",
            "framework": "TensorFlow",
            "inputs": [
                "input_tensor"
            ],
            "outputs": [
                "softmax_tensor",
                "ArgMax",
                "custom"
            ],
            "domain": "Image Recognition",
            "domain_flavour": "",
            "shape": "",
            "trusted": true
        }
    }
]
```

#### Get model's graph
**Path**: `api/model/graph`<br>
**Method**: `GET`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>path</b></td>
            <td>Path to model</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example request**:
```
https://localhost:5000/api/model/graph?path=/path/to/model.pb
```

**Example response**:
```json
{
    "nodes": [
        {
            "id": "input_tensor",
            "label": "Placeholder",
            "properties": {
                "name": "input_tensor",
                "type": "Placeholder"
            },
            "attributes": [
                {
                    "name": "dtype",
                    "attribute_type": "type",
                    "value": "float32"
                }
            ],
            "node_type": "node"
        },
        {
            "id": "node_group_resnet_model",
            "label": "resnet_model",
            "node_type": "group_node",
            "group": "resnet_model"
        },
        {
            "id": "ArgMax",
            "label": "ArgMax",
            "properties": {
                "name": "ArgMax",
                "type": "ArgMax"
            },
            "attributes": [
                {
                    "name": "Tidx",
                    "attribute_type": "type",
                    "value": "int32"
                },
                {
                    "name": "T",
                    "attribute_type": "type",
                    "value": "float32"
                },
                {
                    "name": "output_type",
                    "attribute_type": "type",
                    "value": "int64"
                }
            ],
            "node_type": "node"
        },
        {
            "id": "softmax_tensor",
            "label": "Softmax",
            "properties": {
                "name": "softmax_tensor",
                "type": "Softmax"
            },
            "attributes": [
                {
                    "name": "T",
                    "attribute_type": "type",
                    "value": "float32"
                }
            ],
            "node_type": "node"
        }
    ],
    "edges": [
        {
            "source": "input_tensor",
            "target": "node_group_resnet_model"
        },
        {
            "source": "node_group_resnet_model",
            "target": "ArgMax"
        },
        {
            "source": "node_group_resnet_model",
            "target": "softmax_tensor"
        }
    ]
}
```
### Dictionaries
#### Domains
**Path**: `dict/domains`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "domains": [
        {
            "id": 1,
            "name": "Image Recognition"
        },
        {
            "id": 2,
            "name": "Object Detection"
        },
        {
            "id": 3,
            "name": "Neural Language Processing"
        },
        {
            "id": 4,
            "name": "Recommendation"
        }
    ]
}
```

#### Domain flavours
**Path**: `dict/domain_flavours`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "domain_flavours": [
        {
            "id": 1,
            "name": ""
        },
        {
            "id": 2,
            "name": "SSD"
        },
        {
            "id": 3,
            "name": "Yolo"
        }
    ]
}
```

#### Optimization Types
**Path**: `dict/optimization_types`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "optimization_types": [
        {
            "id": 1,
            "name": "Quantization"
        },
        {
            "id": 2,
            "name": "Graph optimization"
        }
    ]
}
```

#### Optimization Types and support for specific precision
**Path**: `dict/optimization_types/precision`<br>
**Method**: `GET`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>precision</b></td>
            <td>Precision name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "precision": "int8"
}
```
**Example response**:
```json
{
    "optimization_types": [
        {
            "id": 1,
            "name": "Quantization",
            "is_supported": true
        },
        {
            "id": 2,
            "name": "Graph optimization",
            "is_supported": false
        }
    ]
}
```

#### Precisions
**Path**: `dict/precisions`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "precisions": [
        {
            "id": 1,
            "name": "fp32"
        },
        {
            "id": 2,
            "name": "bf16"
        },
        {
            "id": 3,
            "name": "int8"
        }
    ]
}
```

#### Dataloaders

##### Get all dataloaders
**Path**: `dict/dataloaders`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "dataloaders": [
        {
            "id": 1,
            "name": "dummy",
            "help": null,
            "show_dataset_location": false,
            "params": [
                {
                    "name": "shape",
                    "help": null,
                    "value": []
                }
            ],
            "framework": {
                "id": 3,
                "name": "MXNet"
            }
        }
    ]
}
```

##### Get dataloaders for specified framework
**Path**: `dict/dataloaders/framework`<br>
**Method**: `GET`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>framework</b></td>
            <td>Framework name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "framework": "TensorFlow"
}
```
**Example response**:
```json
{
    "dataloaders": [
        {
            "id": 18,
            "name": "CIFAR10",
            "help": null,
            "show_dataset_location": true,
            "params": [
                {
                    "name": "root",
                    "help": null,
                    "value": ""
                },
                {
                    "name": "train",
                    "help": "If True, creates dataset from train subset, otherwise from validation subset",
                    "value": "False"
                }
            ],
            "framework": {
                "id": 1,
                "name": "TensorFlow"
            }
        }
    ]
}
```
#### Transforms

##### Get all transforms
**Path**: `dict/transforms`<br>
**Method**: `GET`<br>
**Example response**:
```json
{
    "transforms": [
        {
            "id": 1,
            "name": "AlignImageChannel",
            "help": null,
            "params": [
                {
                    "name": "dim",
                    "help": "The channel number of result image",
                    "value": ""
                }
            ],
            "framework": {
                "id": 1,
                "name": "TensorFlow"
            }
        }
    ]
}
```

##### Get transforms for specified framework
**Path**: `dict/transforms/framework`<br>
**Method**: `GET`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>framework</b></td>
            <td>Framework name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "framework": "TensorFlow"
}
```
**Example response**:
```json
{
    "transforms": [
        {
            "id": 1,
            "name": "AlignImageChannel",
            "help": null,
            "params": [
                {
                    "name": "dim",
                    "help": "The channel number of result image",
                    "value": ""
                }
            ],
            "framework": {
                "id": 1,
                "name": "TensorFlow"
            }
        }
    ]
}
```

##### Get transforms for specified domain
**Path**: `dict/transforms/domain`<br>
**Method**: `GET`<br>
**Parameters description:**
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>domain</b></td>
            <td>Domain name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "domain": "Object Detection"
}
```
**Example response**:
```json
{
    "transforms": [
        {
            "id": 22,
            "name": "AlignImageChannel",
            "help": null,
            "params": [
                {
                    "name": "dim",
                    "help": "The channel number of result image",
                    "value": ""
                }
            ],
            "framework": {
                "id": 2,
                "name": "ONNXRT"
            }
        }
    ]
}
```

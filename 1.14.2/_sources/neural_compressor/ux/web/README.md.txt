# Intel® Neural Compressor Bench 

## Table of Contents
- [Endpoints](#endpoints)
  - [Project](#project)
    - [Create project](#create-project)
    - [Delete project](#delete-project)
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
    - [Edit optimization](#edit-optimization)
    - [List optimizations](#list-optimizations)
    - [Execute optimization](#execute-optimization)
    - [Pin accuracy benchmark to optimization](#pin-accuracy-benchmark-to-optimization)
    - [Pin performance benchmark to optimization](#pin-performance-benchmark-to-optimization)
  - [Benchmark](#benchmark)
    - [Add benchmark to project](#add-benchmark-to-project)
    - [Get benchmark details](#get-benchmark-details)
    - [Edit benchmark](#edit-benchmark)
    - [List benchmarks](#list-benchmarks)
    - [Execute benchmark](#execute-benchmark)
  - [Profiling](#profiling)
    - [Add profiling to project](#add-profiling-to-project)
    - [Get profiling details](#get-profiling-details)
    - [Edit profiling](#edit-profiling)
    - [Get profiling result as csv](#get-profiling-result-as-csv)
    - [List profilings](#list-profilings)
    - [Execute profiling](#execute-profiling)
  - [Examples](#examples)
    - [List example models](#list-example-models)
    - [Create project from examples](#create-project-from-examples)
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

#### Delete project
**Path**: `api/project/delete`<br>
**Method**: `POST`<br>
**Returns**: id of removed project or null when project not found

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
            <td><b>name</b></td>
            <td>Project name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
    "name": "Project1"
}
```


**Example response**:
```json
{
    "id": 1,
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
    "accuracy_benchmark_id": 1,
    "performance_benchmark_id": null,
    "tuning_details": {
        "id": 1,
        "strategy": "basic",
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_threshold": 0.1,
        "multi_objectives": "performance",
        "exit_policy": {
            "timeout": 0
        },
        "random_seed": 9527,
        "created_at": "2022-01-28 12:37:01",
        "tuning_history": {
            "id": 1,
            "minimal_accuracy": 0.6999299999999999,
            "baseline_accuracy": [
                0.7
            ],
            "baseline_performance": [
                48.57273483276367
            ],
            "last_tune_accuracy": [
                0.706
            ],
            "last_tune_performance": [
                21.251153230667114
            ],
            "best_tune_accuracy": [
                0.706
            ],
            "best_tune_performance": [
                21.251153230667114
            ],
            "history": [
                {
                    "accuracy": [
                        0.699
                    ],
                    "performance": [
                        23.758240222930908
                    ]
                },
                {
                    "accuracy": [
                        0.679
                    ],
                    "performance": [
                        21.93209171295166
                    ]
                },
                {
                    "accuracy": [
                        0.699
                    ],
                    "performance": [
                        21.257209062576294
                    ]
                },
                {
                    "accuracy": [
                        0.679
                    ],
                    "performance": [
                        22.67054533958435
                    ]
                },
                {
                    "accuracy": [
                        0.697
                    ],
                    "performance": [
                        20.60370635986328
                    ]
                },
                {
                    "accuracy": [
                        0.706
                    ],
                    "performance": [
                        21.251153230667114
                    ]
                }
            ]
        }
    }
}
        "modified_at": null
    }
}
```

#### Edit optimization
**Path**: `api/optimization/edit`<br>
**Method**: `POST`

**Parameters description:**
<table border>
    <thead>
        <tr>
            <th colspan=3>Parameter</th>
            <th>Description</th>
            <th>Required</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=3><b>id</b></td>
            <td>Optimization id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td colspan=3><b>precision_id</b></td>
            <td>Precision id</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=3><b>dataset_id</b></td>
            <td>Dataset id</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td rowspan=9><b>tuning_details</b></td>
            <td colspan=2><b>batch_size</b></td>
            <td>Optimization batch size</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>sampling_size</b></td>
            <td>Optimization sampling size</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>multi_objectives</b></td>
            <td>Optimization objective</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>strategy</b></td>
            <td>Optimization strategy</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>accuracy_criterion_type</b></td>
            <td>Optimization accuracy criterion type</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>accuracy_criterion_threshold</b></td>
            <td>Optimization accuracy criterion threshold</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td colspan=2><b>random_seed</b></td>
            <td>Optimization random seed</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td rowspan=2><b>exit_policy</b></td>
            <td><b>timeout</b></td>
            <td>Optimization timeout</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td><b>max_trails</b></td>
            <td>Optimization max trials</td>
            <td align="center">&#x2610;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
    "tuning_details": {
        "batch_size": 123,
        "sampling_size": 456,
        "strategy": "mse",
        "accuracy_criterion_type": "absolute",
        "accuracy_criterion_threshold": 0.23,
        "multi_objectives": "performance",
        "exit_policy": {
            "timeout": 60,
            "max_trials": 125
        },
        "random_seed": 95270,
    }
}
```

**Example response**:
```json
{
    "id": 1,
    "tuning_details": {
        "id": 1,
        "strategy": "mse",
        "accuracy_criterion_type": "absolute",
        "accuracy_criterion_threshold": 0.23,
        "multi_objectives": "performance",
        "exit_policy": {
            "timeout": "60",
            "max_trials": 125
        },
        "random_seed": 95270,
        "created_at": "2022-08-09 15:54:44",
        "modified_at": "2022-08-09 16:43:06"
    },
    "batch_size": 123,
    "sampling_size": 456
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
            "accuracy_benchmark_id": 1,
            "performance_benchmark_id": 2,
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
            "accuracy_benchmark_id": null,
            "performance_benchmark_id": null,
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
        "message": "started",
        "request_id": "asd",
        "size_input_model": 16,
        "config_path": "/path/to/config.yaml",
        "output_path": "/path/to/output.txt"
    }
]
```

**Tuning history** <details><br><br>


```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": null,
        "last_tune_performance": null,
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [],
    }
]
```

```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": [
            0.699
        ],
        "last_tune_performance": [
            23.758240222930908
        ],
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    23.758240222930908
                ]
            }
        ]
    }
]
```

```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": [
            0.679
        ],
        "last_tune_performance": [
            21.93209171295166
        ],
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    23.758240222930908
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    21.93209171295166
                ]
            }
        ]
    }
]
```

```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": [
            0.699
        ],
        "last_tune_performance": [
            21.257209062576294
        ],
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    23.758240222930908
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    21.93209171295166
                ]
            },
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    21.257209062576294
                ]
            }
        ]
    }
]
```

```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": [
            0.679
        ],
        "last_tune_performance": [
            22.67054533958435
        ],
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    23.758240222930908
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    21.93209171295166
                ]
            },
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    21.257209062576294
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    22.67054533958435
                ]
            }
        ]
    }
]
```

```json
[
    "tuning_history",
    {
        "request_id": "asd",
        "optimization_id": 1,
        "accuracy_criterion_type": "relative",
        "accuracy_criterion_value": 0.0001,
        "minimal_accuracy": 0.6999299999999999,
        "baseline_accuracy": [
            0.7
        ],
        "baseline_performance": [
            48.57273483276367
        ],
        "last_tune_accuracy": [
            0.697
        ],
        "last_tune_performance": [
            20.60370635986328
        ],
        "best_tune_accuracy": null,
        "best_tune_performance": null,
        "history": [
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    23.758240222930908
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    21.93209171295166
                ]
            },
            {
                "accuracy": [
                    0.699
                ],
                "performance": [
                    21.257209062576294
                ]
            },
            {
                "accuracy": [
                    0.679
                ],
                "performance": [
                    22.67054533958435
                ]
            },
            {
                "accuracy": [
                    0.697
                ],
                "performance": [
                    20.60370635986328
                ]
            }
        ]
    }
]
```

</details>

```json
[
    "optimization_finish",
    {
        "request_id": "asd",
        "project_id": 1,
        "id": 3,
        "name": "Optimization3",
        "precision": {
            "id": 3,
            "name": "int8"
        },
        "optimization_type": {
            "id": 1,
            "name": "Quantization"
        },
        "dataset": {
            "id": 2,
            "name": "Dataset2"
        },
        "config_path": "/path/to/config.yaml",
        "log_path": "/path/to/output.txt",
        "execution_command": "python /path/to/tune_model.py --input-graph /path/to/model.pb --output-graph /path/to/optimized_model.pb --config /path/to/config.yaml --framework tensorflow",
        "batch_size": 100,
        "sampling_size": 100,
        "status": "success",
        "created_at": "2022-03-14 17:52:07",
        "last_run_at": "2022-03-14 18:22:40",
        "duration": 1231,
        "optimized_model": {
            "id": 4,
            "name": "optimization3",
            "path": "/path/to/optimized_model.pb",
            "framework": {
                "id": 1,
                "name": "TensorFlow"
            },
            "size": 4,
            "precision": {
                "id": 3,
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
            "input_nodes": [],
            "output_nodes": [],
            "supports_profiling": true,
            "supports_graph": true,
            "created_at": "2022-03-14 18:22:40"
        },
        "accuracy_benchmark_id": null,
        "performance_benchmark_id": null,
        "tuning_details": {
            "id": 3,
            "strategy": "basic",
            "accuracy_criterion_type": "relative",
            "accuracy_criterion_threshold": 0.0001,
            "multi_objectives": "performance",
            "exit_policy": {
                "timeout": 0
            },
            "random_seed": 9527,
            "created_at": "2022-03-14 17:52:07",
            "modified_at": "2022-03-14 18:22:40",
            "tuning_history": {
                "id": 3,
                "minimal_accuracy": 0.6999299999999999,
                "baseline_accuracy": [
                    0.7
                ],
                "baseline_performance": [
                    48.57273483276367
                ],
                "last_tune_accuracy": [
                    0.706
                ],
                "last_tune_performance": [
                    21.251153230667114
                ],
                "best_tune_accuracy": [
                    0.706
                ],
                "best_tune_performance": [
                    21.251153230667114
                ],
                "history": [
                    {
                        "accuracy": [
                            0.699
                        ],
                        "performance": [
                            23.758240222930908
                        ]
                    },
                    {
                        "accuracy": [
                            0.679
                        ],
                        "performance": [
                            21.93209171295166
                        ]
                    },
                    {
                        "accuracy": [
                            0.699
                        ],
                        "performance": [
                            21.257209062576294
                        ]
                    },
                    {
                        "accuracy": [
                            0.679
                        ],
                        "performance": [
                            22.67054533958435
                        ]
                    },
                    {
                        "accuracy": [
                            0.697
                        ],
                        "performance": [
                            20.60370635986328
                        ]
                    },
                    {
                        "accuracy": [
                            0.706
                        ],
                        "performance": [
                            21.251153230667114
                        ]
                    }
                ]
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

#### Edit benchmark
**Path**: `api/benchmark/edit`<br>
**Method**: `POST`

**Parameters description:**
<table border>
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
        <tr>
            <td><b>dataset_id</b></td>
            <td>Dataset id</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td><b>mode</b></td>
            <td>Benchmark mode</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td><b>batch_size</b></td>
            <td>Benchmark batch size</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td><b>number_of_instance</b></td>
            <td>Optimization number of instance</td>
            <td align="center">&#x2610;</td>
        </tr>
        <tr>
            <td><b>cores_per_instance</b></td>
            <td>Optimization cores per instance</td>
            <td align="center">&#x2610;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
    "dataset_id": 4,
    "mode": "accuracy",
    "batch_size": 64,
    "cores_per_instance": 7,
    "number_of_instance": 4
}
```

**Example response**:
```json
{
    "id": 1,
    "dataset_id": 4,
    "batch_size": 64,
    "mode": "accuracy",
    "cores_per_instance": 7,
    "number_of_instance": 4
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

#### Edit profiling
**Path**: `api/profiling/edit`<br>
**Method**: `POST`

**Parameters description:**
<table border>
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
        <tr>
            <td><b>dataset_id</b></td>
            <td>Dataset id</td>
            <td align="center">&#x2610;</td>
        </tr>
 
    </tbody>
</table>

**Example body**:
```json
{
    "id": 1,
    "dataset_id": 4,
    "num_threads": 14
}
```

**Example response**:
```json
{
    "id": 1,
    "dataset_id": 4,
    "num_threads": 14
}
```

#### Get profiling result as csv
**Path**: `api/profiling/results/csv`<br>
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
```
node_name,total_execution_time,accelerator_execution_time,cpu_execution_time,op_run,op_defined
Node_A,123,0,123,117,140
Node_B,456,0,456,3,3
Node_C,234,0,234,12,12
Node_D,304,0,304,5,185
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

### Examples
#### List example models
**Path**: `api/examples/list`<br>
**Method**: `GET`

**Example response**:

```json
[
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "inception_v3"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "inception_v4"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "mobilenetv1"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "resnet50_v1_5"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "resnet101"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "faster_rcnn_inception_resnet_v2"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "faster_rcnn_resnet101"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "mask_rcnn_inception_v2"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "ssd_mobilenet_v1"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "ssd_resnet50_v1"
    }
]
```

#### Create project from examples
**Path**: `api/examples/add`<br>
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
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>New project name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>framework</b></td>
            <td>Framework of example model</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>domain</b></td>
            <td>Domain of example model</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>model</b></td>
            <td>Example model name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>progress_steps</b></td>
            <td>Number of steps to show in model's percentage download progress</td>
            <td align="center">&#x2610;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "abc",
    "name": "Predefined model test 2",
    "framework": "TensorFlow",
    "domain": "Image Recognition",
    "model": "inception_v3",
    "progress_steps": 4
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
    "create_example_project_start",
    {
        "status": "success",
        "data": {
            "message": "Creating project from examples.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Downloading example model.",
            "request_id": "abc"
        }
    }
]
```
```json
[
    "download_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "request_id": "abc",
            "url": "https://url/to/model.pb"
        }
    }
]
```

```json
[
    "download_progress",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "progress": 25
        }
    }
]
```

```json
[
    "download_progress",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "progress": 50
        }
    }
]
```

```json
[
    "download_progress",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "progress": 75
        }
    }
]
```

```json
[
    "download_progress",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "progress": 100
        }
    }
]
```

```json
[
    "download_finish",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "path": "/path/to/model.pb"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Adding project for example model.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Adding example model to project.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Adding dummy dataset to project.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Adding optimization to project.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_progress",
    {
        "status": "success",
        "data": {
            "message": "Adding benchmark to project.",
            "request_id": "abc"
        }
    }
]
```

```json
[
    "create_example_project_finish",
    {
        "status": "success",
        "data": {
            "message": "Example project has been added.",
            "request_id": "abc",
            "project_id": 1,
            "model_id": 1,
            "dataset_id": 1,
            "optimization_id": 1,
            "benchmark_id": 1
        }
    }
]
```

### Examples
#### List example models
**Path**: `api/examples/list`<br>
**Method**: `GET`

**Example response**:

```json
[
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "inception_v3"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "inception_v4"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "mobilenetv1"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "resnet50_v1_5"
    },
    {
        "domain": "Image Recognition",
        "framework": "TensorFlow",
        "model": "resnet101"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "faster_rcnn_inception_resnet_v2"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "faster_rcnn_resnet101"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "mask_rcnn_inception_v2"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "ssd_mobilenet_v1"
    },
    {
        "domain": "Object Detection",
        "framework": "TensorFlow",
        "model": "ssd_resnet50_v1"
    }
]
```

#### Create project from examples
**Path**: `api/examples/add`<br>
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
            <td><b>request_id</b></td>
            <td>Request id</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>name</b></td>
            <td>New project name</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>framework</b></td>
            <td>Framework of example model</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>domain</b></td>
            <td>Domain of example model</td>
            <td align="center">&#x2611;</td>
        </tr>
        <tr>
            <td><b>model</b></td>
            <td>Example model name</td>
            <td align="center">&#x2611;</td>
        </tr>
    </tbody>
</table>

**Example body**:
```json
{
    "request_id": "abc",
    "name": "Predefined model test 2",
    "framework": "TensorFlow",
    "domain": "Image Recognition",
    "model": "inception_v3"
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
    "download_start",
    {
        "status": "success",
        "data": {
            "message": "started",
            "request_id": "abc",
            "url": "https://url.to/model.pb"
        }
    }
]
```

```json
[
    "download_finish",
    {
        "status": "success",
        "data": {
            "request_id": "abc",
            "path": "/path/to/model.pb"
        }
    }
]
```

```json
[
    "example_finish",
    {
      "status": "success",
      "data": {
        "message": "Example project has been added.",
        "request_id": "abc",
        "project_id": 1,
        "model_id": 1,
        "dataset_id": 1,
        "optimization_id": 1,
        "benchmark_id": 1
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
**Path**: `api/dict/domains`<br>
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
**Path**: `api/dict/domain_flavours`<br>
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
**Path**: `api/dict/optimization_types`<br>
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
**Path**: `api/dict/optimization_types/precision`<br>
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
**Path**: `api/dict/precisions`<br>
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
**Path**: `api/dict/dataloaders`<br>
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
**Path**: `api/dict/dataloaders/framework`<br>
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
**Path**: `api/dict/transforms`<br>
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
**Path**: `api/dict/transforms/framework`<br>
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
**Path**: `api/dict/transforms/domain`<br>
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

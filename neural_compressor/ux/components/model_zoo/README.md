# Examples endpoints

## Table of contents
- [List models](#list-models)
- [Download model](#download-model)
- [Download config](#download-config)

## List models
### Path
`/api/list_model_zoo`

### Example response

```json
[
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "inception_v3",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "inception_v4",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "mobilenetv1",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "resnet50_v1_5",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "resnet101",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "faster_rcnn_inception_resnet_v2",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "faster_rcnn_resnet101",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "mask_rcnn_inception_v2",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "ssd_mobilenet_v1",
        "model_path": "",
        "yaml": ""
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "ssd_resnet50_v1",
        "model_path": "",
        "yaml": ""
    }
]
```

## Download model

Endpoint that enable downloading models for Examples.

It will create new directory: `${workspace_path}/examples/${framework}/${domain}/${model}` and put downloaded there. 

### Path
`/api/download_model`

### Request Body

| **Parameter** | **Description** |
|:----------|:------------|
| **id** | workload ID | True
| **workspace_path** | path to current UX workspace |
| **framework** | Workload framework name |
| **domain** | Model's domain name |
| **model** | Model name |
| **progress_steps** | Number of reporting steps of model download.<br><br>If not provided only start and finish download event will be sent. |

#### Example body
```json
{
    "id": "1",
    "workspace_path": "/home/user",
    "framework": "tensorflow",
    "domain": "image_recognition",
    "model": "inception_v3",
    "progress_steps": 5
}
```

## Download config

Endpoint that enable downloading config for models from Examples.

It will create new directory: `${workspace_path}/examples/${framework}/${domain}/${model}` and put downloaded config there. 

### Path
`/api/download_config`

### Request Body

| **Parameter** | **Description** |
|:----------|:------------|
| **id** | workload ID | True
| **workspace_path** | path to current UX workspace |
| **framework** | Workload framework name |
| **domain** | Model's domain name |
| **model** | Model name |
| **progress_steps** | Number of reporting steps of model download.<br><br>If not provided only start and finish download event will be sent. |

#### Example body
```json
{
    "id": "1",
    "workspace_path": "/home/user",
    "framework": "tensorflow",
    "domain": "image_recognition",
    "model": "inception_v3",
    "progress_steps": 5
}
```
# Model Zoo endpoints

## Table of contents
- [List models](#list-models)
- [Download model](#download-model)
- [Download config](#download-config)

## List models
### Path
`/api/list_model_zoo`

### Example responses

#### Case when `LPOT_REPOSITORY_PATH` is set to `/home/user/lpot`

```json
[
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "inception_v3",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/image_recognition/inceptionv3.yaml"
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "inception_v4",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/image_recognition/inceptionv4.yaml"
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "mobilenetv1",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/image_recognition/mobilenet_v1.yaml"
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "resnet50_v1_5",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/image_recognition/resnet50_v1_5.yaml"
    },
    {
        "domain": "image_recognition",
        "framework": "tensorflow",
        "model": "resnet101",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/image_recognition/resnet101.yaml"
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "faster_rcnn_inception_resnet_v2",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/object_detection/faster_rcnn_inception_resnet_v2.yaml"
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "faster_rcnn_resnet101",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/object_detection/faster_rcnn_resnet101.yaml"
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "mask_rcnn_inception_v2",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/object_detection/mask_rcnn_inception_v2.yaml"
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "ssd_mobilenet_v1",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/object_detection/ssd_mobilenet_v1.yaml"
    },
    {
        "domain": "object_detection",
        "framework": "tensorflow",
        "model": "ssd_resnet50_v1",
        "model_path": "",
        "yaml": "/home/user/lpot/examples/tensorflow/object_detection/ssd_resnet50_v1.yaml"
    }
]
```

#### Case when `LPOT_REPOSITORY_PATH` is not set
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

Endpoint that enable downloading models for Model Zoo.

It will create new directory: `${workspace_path}/model_zoo/${framework}/${domain}/${model}` and put downloaded there. 

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

Endpoint that enable downloading config for models from Model Zoo.

It will create new directory: `${workspace_path}/model_zoo/${framework}/${domain}/${model}` and put downloaded config there. 

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
# Workload class

*Workload* class represents single workload.

To create workload there should be passed dictionary with `framework`, `domain`, `model_name`, `accuracy_goal` values and paths to the UX workspace, dataset and model.

Example init data:  
```json
{
    "framework": "tensorflow",
    "domain": "image_recognition",
    "workspace_path": "/user/home/my_workspace",
    "dataset_path": "/user/home/dataset",
    "accuracy_goal": 0.1,
    "model_path": "/user/home/my_workspace/my_model.pb",
    "model_name": "my_model"
}
```

While creating workload object it search for predefined config for specified **framework** and **domain** and updates such predefined config with passed values.
```python
from lpot.ux.utils.workload.workload import Workload

workload = Workload({
    "framework": "tensorflow",
    "domain": "image_recognition",
    "workspace_path": "/user/home/my_workspace",
    "dataset_path": "/user/home/dataset",
    "accuracy_goal": 0.1,
    "model_path": "/user/home/my_workspace/my_model.pb",
    "model_name": "my_model"
})
``` 

*Workload* class is serializable by `.serialize()` method. It removes unset parameters.

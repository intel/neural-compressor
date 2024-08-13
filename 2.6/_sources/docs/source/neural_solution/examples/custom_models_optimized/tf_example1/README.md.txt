
## An end-to-end example: quantize a custom model with Neural Solution

In this example, we show how to quantize a [custom model](https://github.com/intel/neural-compressor/tree/master/examples/helloworld/tf_example1) with Neural Solution.

### Objective
- Demonstrate how to prepare requirements.
- Demonstrate how to start the Neural Solution Service.
- Demonstrate how to prepare an optimization task request and submit it to Neural Solution Service.
- Demonstrate how to query the status of the task and fetch the optimization result.
- Demonstrate how to query and manage the resource of the cluster.

### Requirements
Customizing the model requires preparing the following folders and files.
1. dataset/, place dataset
2. model/, place model weight and configuration files
3. run.py, the running python script

The folder structure is as follows:
```shell
├── dataset
│   └── train-00173-of-01024
├── model
│   └── mobilenet_v1_1.0_224_frozen.pb
├── README.html
├── task_request_distributed.json
├── task_request.json
└── test.py
```

### Start the Neural Solution Service

```shell
# Activate your environment
conda activate ENV

# Start neural solution service with default configuration, log will be saved in the "serve_log" folder.
neural_solution start

# Start neural solution service with custom configuration
neural_solution start --task_monitor_port=22222 --result_monitor_port=33333 --restful_api_port=8001

# Stop neural solution service with default configuration
neural_solution stop

# Help Manual
neural_solution -h
# Help output

usage: neural_solution {start,stop} [-h] [--hostfile HOSTFILE] [--restful_api_port RESTFUL_API_PORT] [--grpc_api_port GRPC_API_PORT]
                   [--result_monitor_port RESULT_MONITOR_PORT] [--task_monitor_port TASK_MONITOR_PORT] [--api_type API_TYPE]
                   [--workspace WORKSPACE] [--conda_env CONDA_ENV] [--upload_path UPLOAD_PATH] [--query] [--join JOIN] [--remove REMOVE]

Neural Solution

positional arguments:
  {start,stop,cluster}  start/stop/management service

optional arguments:
  -h, --help            show this help message and exit
  --hostfile HOSTFILE   start backend serve host file which contains all available nodes
  --restful_api_port RESTFUL_API_PORT
                        start restful serve with {restful_api_port}, default 8000
  --grpc_api_port GRPC_API_PORT
                        start gRPC with {restful_api_port}, default 8000
  --result_monitor_port RESULT_MONITOR_PORT
                        start serve for result monitor at {result_monitor_port}, default 3333
  --task_monitor_port TASK_MONITOR_PORT
                        start serve for task monitor at {task_monitor_port}, default 2222
  --api_type API_TYPE   start web serve with all/grpc/restful, default all
  --workspace WORKSPACE
                        neural solution workspace, default "./ns_workspace"
  --conda_env CONDA_ENV
                        specify the running environment for the task
  --upload_path UPLOAD_PATH
                        specify the file path for the tasks
  --query               [cluster parameter] query cluster information
  --join JOIN           [cluster parameter] add new node into cluster
  --remove REMOVE       [cluster parameter] remove <node-id> from cluster
```


### Submit optimization task

- Step 1: Prepare the json file includes request content. In this example, we have created request that quantize a [custom model](https://github.com/intel/neural-compressor/tree/master/examples/helloworld/tf_example1).

```shell
[user@server tf_example1]$ cd path/to/neural_solution/neural_solution/examples/custom_models_optimized/tf_example1
[user@server tf_example1]$ cat task_request.json
{
    "script_url": "tf_example1",
    "optimized": "True",
    "arguments": [
        "--dataset_location=dataset", "--model_path=model"
    ],
    "approach": "static",
    "requirements": [
    ],
    "workers": 1
}
```
When using distributed quantization, the `workers` needs to be set to greater than 1 when submitting a request.
```shell
[user@server tf_example1]$ cat task_request_distributed.json
{
    "script_url": "tf_example1",
    "optimized": "True",
    "arguments": [
        "--dataset_location=dataset", "--model_path=model"
    ],
    "approach": "static",
    "requirements": [
    ],
    "workers": 3
}
```


- Step 2: Submit the task request to service, and it will return the submit status and task id for future use.

```shell
[user@server tf_example1]$ curl -H "Content-Type: application/json" --data @./task.json  http://localhost:8000/task/submit/

# response if submit successfully
{
    "status": "successfully",
    "task_id": "7602cd63d4c849e7a686a8165a77f69d",
    "msg": "Task submitted successfully"
}
```



### Query optimization result

- Query the task status and result according to the `task_id`.

``` shell
[user@server tf_example1]$ curl -X GET  http://localhost:8000/task/status/{task_id}
# return the task status
{
    "status": "done",
    "tuning_info": {},
    "optimization_result": {
        "optimization time (seconds)": "151.16",
        "Accuracy": "0.8617",
        "Duration (seconds)": "17.8213",
        "result_path": "http://localhost:8000/download/7602cd63d4c849e7a686a8165a77f69d"
    }
}

```
### Download optimized model

- Download the optimized model according to the `task_id`.

``` shell
[user@server tf_example1]$ curl -X GET  http://localhost:8000/download/{task_id} --output quantized_model.zip
# download quantized_model.zip
```

### Manage resource
```shell
# query cluster information
neural_solution cluster --query

# add new node into cluster
# parameter: "<node1> <number_of_sockets> <number_of_threads>;<node2> <number_of_sockets> <number_of_threads>"
neural_solution cluster --join "host1 2 20; host2 5 20"

# remove node from cluster according to id
neural_solution cluster --remove <node-id>
```

### Stop the service
```shell
neural_solution stop
```

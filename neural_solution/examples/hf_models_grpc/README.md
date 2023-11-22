## An end-to-end example: quantize a Hugging Face model with Neural Solution gRPC API

In this example, we show how to quantize a Hugging Face model with Neural Solution gRPC API.

### Objective
- Demonstrate how to start the Neural Solution Service.
- Demonstrate how to prepare an optimization task request and submit it to Neural Solution Service.
- Demonstrate how to query the status of the task and fetch the optimization result.


### Start the Neural Solution Service

```shell
# Activate your environment
conda activate ENV

# Start neural solution service with default configuration, log will be saved in the "serve_log" folder.
neural_solution start

# Start neural solution service with custom configuration
neural_solution start --task_monitor_port=22222 --result_monitor_port=33333 --grpc_api_port=8001 --api_type=grpc

# Stop neural solution service with default configuration
neural_solution stop

# Help Manual
neural_solution -h
# Help output

usage: neural_solution {start,stop} [-h] [--hostfile HOSTFILE] [--restful_api_port RESTFUL_API_PORT] [--grpc_api_port GRPC_API_PORT]
                   [--result_monitor_port RESULT_MONITOR_PORT] [--task_monitor_port TASK_MONITOR_PORT] [--api_type API_TYPE]
                   [--workspace WORKSPACE] [--conda_env CONDA_ENV] [--upload_path UPLOAD_PATH]

Neural Solution

positional arguments:
  {start,stop}          start/stop service

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
```


### Submit optimization task

- Step 1: Prepare the json file includes request content. In this example, we have created request that quantize a [Text classification model](https://github.com/huggingface/transformers/tree/v4.21-release/examples/pytorch/text-classification) from Hugging Face.

```shell
[user@server hf_models_grpc]$ cd path/to/neural_solution/examples/hf_models_grpc
[user@server hf_models_grpc]$ cat task_request.json
{
    "script_url": "https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py",
    "optimized": "False",
    "arguments": [
        "--model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result"
    ],
    "approach": "static",
    "requirements": [],
    "workers": 1
}
```


- Step 2: Submit the task request to service, and it will return the submit status and task id for future use.

```shell
[user@server hf_models_grpc]$ python client.py submit --request="test_task_request.json"

# response if submit successfully
2023-06-20 14:34:55 [INFO] Neural Solution is running.
2023-06-20 14:34:55 [INFO] successfully
2023-06-20 14:34:55 [INFO] d3e10a49326449fb9d0d62f2bfc1cb43
2023-06-20 14:34:55 [INFO] Task submitted successfully
```



### Query optimization result

- Query the task status and result according to the `task_id`.

``` shell
[user@server hf_models_grpc]$ python client.py --task_monitor_port=22222 --result_monitor_port=33333 --grpc_api_port=8001 query --task_id="d3e10a49326449fb9d0d62f2bfc1cb43"


```
### Stop the service
```shell
neural_solution stop
```

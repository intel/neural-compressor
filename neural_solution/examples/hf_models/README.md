## An end-to-end example: quantize a Hugging Face model with Neural Solution

In this example, we show how to quantize a Hugging Face model with Neural Solution.

### Objective
- Demonstrate how to start the Neural Solution Service.
- Demonstrate how to prepare an optimization task request and submit it to Neural Solution Service.
- Demonstrate how to query the status of the task and fetch the optimization result.


### Start the Neural Solution Service

```shell
# Activate your environment
conda activate ENV

# Start neural solution service with default configuration, log will be saved in the "serve_log" folder.
ns start

# Start neural solution service with custom configuration
ns start --task_monitor_port=22222 --result_monitor_port=33333 --restful_api_port=8001

# Stop neural solution service with default configuration
ns stop

# Help Manual
ns help
# Help output

 *** usage: ns {start|stop} ***
     start      : start serve
     stop       : stop serve

  more start parameters: [usage: ns start {--parameter=value}] [e.g. --restful_api_port=8000]
    --hostfile           : start backend serve host file which contains all available nodes
    --restful_api_port         : start web serve with {restful_api_port}, defult 8000
    --api_type           : start web serve with grpc/http, defult http
    --task_monitor_port  : start serve for task monitor at {task_monitor_port}, defult 2222
    --result_monitor_port: start serve for result monitor at {result_monitor_port}, defult 3333
    --workspace          : neural solution workspace, defult "./"
    --conda_env          : specify the running environment for the task
    --upload_path        : specify the file path for the tasks
```


### Submit optimization task

- Step 1: Prepare the json file includes request content. In this example, we have created request that quantize a [Text classification model](https://github.com/huggingface/transformers/tree/v4.21-release/examples/pytorch/text-classification) from Hugging Face.

```shell
[user@server hf_model]$ cd path/to/neural_solution/examples/hf_model
[user@server hf_model]$ cat task_request.json
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
[user@server hf_model]$ curl -H "Content-Type: application/json" --data @./task.json  http://localhost:8000/task/submit/

# response if submit successfully
{
    "status": "successfully",
    "task_id": "cdf419910f9b4d2a8320d0e420ac1d0a",
    "msg": "Task submitted successfully"
}
```



### Query optimization result

- Query the task status and result according to the `task_id`.

``` shell
[user@server hf_model]$ curl  -X GET  http://localhost:8000/task/status/{task_id}

# return the task status
{
    "status": "done",
    "optimized_result": {
        "optimization time (seconds)": "58.15",
        "accuracy": "0.3162",
        "duration (seconds)": "4.6488"
    },
    "result_path": "/path/to/projects/neural solution service/workspace/fafdcd3b22004a36bc60e92ec1d646d0/q_model_path"
}

```
### Stop the serve
```shell
ns stop
```
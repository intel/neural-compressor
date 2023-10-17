# Get started

- [Get started](#get-started)
  - [Install Neural Solution](#install-neural-solution)
    - [Prerequisites](#prerequisites)
    - [Method 1. Using pip](#method-1-using-pip)
    - [Method 2. Building from source](#method-2-building-from-source)
  - [Start service](#start-service)
  - [Submit task](#submit-task)
  - [Query task status](#query-task-status)
  - [Stop service](#stop-service)
  - [Inspect logs](#inspect-logs)
  - [Manage resource](#manage-resource)
    - [Node States](#node-states)
    - [Query cluster](#query-cluster)
    - [Add node](#add-node)
    - [Remove node](#remove-node)

## Install Neural Solution
### Prerequisites
- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/)
- Install [Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build)
- Python 3.8 or later

There are two ways to install the neural solution:
### Method 1. Using pip
```
pip install neural-solution
```
### Method 2. Building from source

```shell
# get source code
git clone https://github.com/intel/neural-compressor
cd neural-compressor

# install neural compressor
pip install -r requirements.txt
python setup.py install

# install neural solution
pip install -r neural_solution/requirements.txt
python setup.py neural_solution install
```

## Start service

```shell
# Start neural solution service with custom configuration
neural_solution start --task_monitor_port=22222 --result_monitor_port=33333 --restful_api_port=8001

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

## Submit task

- For RESTful API: `[user@server hf_model]$ curl -H "Content-Type: application/json" --data @./task.json  http://localhost:8000/task/submit/`
- For gRPC API: `python  -m neural_solution.frontend.gRPC.client submit --request="test.json"`

> For more details, please reference the [API description](./description_api.html) and [examples](../../examples/README.html).

## Query task status

Query the task status and result according to the `task_id`.

- For RESTful API: `[user@server hf_model]$ curl  -X GET  http://localhost:8000/task/status/{task_id}`
- For gRPC API: `python  -m neural_solution.frontend.gRPC.client query --task_id={task_id}`

> For more details, please reference the [API description](./description_api.html) and [examples](../../examples/README.html).

## Stop service

```shell
# Stop neural solution service with default configuration
neural_solution stop
```

## Inspect logs

The default logs locate in `./ns_workspace/`. Users can specify a custom workspace by using `neural_solution ---workspace=/path/to/custom/workspace`.

There are several logs under workspace:

```shell
(ns) [username@servers ns_workspace]$ tree
.
├── db
│   └── task.db # database to save the task-related information
├── serve_log # service running log
│   ├── backend.log # backend log 
│   ├── frontend_grpc.log # grpc frontend log
│   └── frontend.log # HTTP/RESTful frontend log
├── task_log # overall log for each task
│   ├── task_bdf0bd1b2cc14bc19bce12d4f9b333c7.txt # task log
│   └── ...
└── task_workspace # the log for each task
    ...
    ├── bdf0bd1b2cc14bc19bce12d4f9b333c7 # task_id
    ...

```

## Manage resource
Neural Solution supports cluster management for service maintainers, providing several command-line tools for efficient resource management. 

### Node States

Each node in the cluster can have three different states:

- Alive: Represents a node that is functioning properly and available to handle requests.
- Join: Indicates that a node is in the process of being added to the cluster but has not fully joined yet.
- Remove: Indicates that a node is scheduled to be removed from the cluster.

Below are some commonly used commands and their usage:

### Query cluster
This command is used to query the current status of the cluster. No additional parameters are required, simply enter the following command:
```shell
neural_solution cluster --query
```
### Add node
This command is used to add nodes to the cluster. You can either specify a host file or provide a list of nodes separated by ";". The node format consists of three parts: hostname, number_of_sockets, and cores_per_socket. Here's a breakdown of each part:

- hostname: This refers to the name or IP address of the node that you want to add to the cluster. It identifies the specific machine or server that will be part of the cluster.

- number_of_sockets: This indicates the number of physical CPU sockets available on the node. A socket is a physical component that houses one or more CPU cores. It represents a physical processor unit.

- cores_per_socket: This specifies the number of CPU cores present in each socket. A core is an individual processing unit within a CPU.

For example:
```shell
neural_solution cluster --join "host1 2 20; host2 4 20"
```
### Remove node
This command is used to remove nodes from the cluster based on the IDs obtained from the query. The IDs can be passed as a parameter to the command. For example:
```shell
neural_solution cluster --remove <query_id>
```
Please note that the above commands are just examples and may require additional parameters or configurations based on your specific setup.

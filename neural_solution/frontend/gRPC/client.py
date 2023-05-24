# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client of gRPC frontend."""

import grpc
from neural_solution.frontend.gRPC.proto import (
    neural_solution_pb2,
    neural_solution_pb2_grpc
)

from neural_solution.config import config

def run():
    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel('localhost:' + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    # Ping serve
    request = neural_solution_pb2.EmptyRequest()
    response = stub.Ping(request)
    print(response.status)
    print(response.msg)

    # Create a task request with the desired fields
    request = neural_solution_pb2.Task(
        script_url='https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py',
        optimized=False,
        arguments=["--model_name_or_path", "bert-base-cased",  "--task_name mrpc", "--do_eval", "--output_dir" "result"],
        approach="approach",
        requirements=[],
        workers=1
    )

    # Call the SubmitTask RPC on the server
    response = stub.SubmitTask(request)

    # Process the response
    print(response.status)
    print(response.task_id)
    print(response.msg)


def run_query_task_result(task_id="38e459ef31bc41ddbb4228004d7b2979"):
    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel('localhost:' + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    request = neural_solution_pb2.TaskId(task_id=task_id)
    response = stub.QueryTaskResult(request)
    print(response.status)
    print(response.tuning_information)
    print(response.optimization_result)

def run_query_task_status(task_id):
    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel('localhost:' + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    request = neural_solution_pb2.TaskId(task_id=task_id)
    response = stub.GetTaskById(request)
    print(response.status)
    print(response.optimized_result)
    print(response.result_path)




if __name__ == '__main__':
    run_query_task_result()
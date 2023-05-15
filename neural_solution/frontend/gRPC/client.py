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

def run():
    # Create a gRPC channel
    port = '50051'
    channel = grpc.insecure_channel('localhost:' + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    # Create a task request with the desired fields
    request = neural_solution_pb2.Task(
        script_url='https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py',
        optimized=False,
        arguments=["--model_name_or_path", "bert-base-cased",  "--task_name mrpc", "--do_eval", "--output_dir" "result"],
        approach="approach",
        requirements=[],
        workers=5
    )

    # Call the SubmitTask RPC on the server
    response = stub.SubmitTask(request)

    # Process the response
    print(response.status)
    print(response.task_id)
    print(response.msg)


if __name__ == '__main__':
    run()
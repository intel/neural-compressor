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

"""Server of gRPC frontend."""


from concurrent import futures
import logging
import grpc

from neural_solution.frontend.gRPC.proto import (
    neural_solution_pb2,
    neural_solution_pb2_grpc)

from neural_solution.frontend.utility import submit_task_to_db


class TaskSubmitterServicer(neural_solution_pb2_grpc.TaskServiceServicer):
    def __init__(self) -> None:
        pass

    def SubmitTask(self, task, context):
        # Process the task
        print(f"Submit task to task db")
        result = submit_task_to_db(task)
        # Return a response
        response = neural_solution_pb2.TaskResponse(**result)
        return response


def serve():
    port = '50051' # TODO exposed it to user
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    neural_solution_pb2_grpc.add_TaskServiceServicer_to_server(
        TaskSubmitterServicer(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()

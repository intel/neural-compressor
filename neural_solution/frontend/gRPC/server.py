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
import argparse

from neural_solution.frontend.gRPC.proto import (
    neural_solution_pb2,
    neural_solution_pb2_grpc)

from neural_solution.config import config
from neural_solution.utility import get_db_path
from neural_solution.frontend.task_submitter import task_submitter

from neural_solution.frontend.utility import (
    submit_task_to_db,
    check_service_status,
    query_task_status,
    query_task_result)

from neural_solution.utility import dict_to_str
class TaskSubmitterServicer(neural_solution_pb2_grpc.TaskServiceServicer):
    def __init__(self) -> None:
        pass

    def Ping(self, empty_msg, context):
        print(f"Ping grpc serve.")
        port_lst = [config.result_monitor_port]
        result = check_service_status(port_lst, service_address=config.service_address)
        response = neural_solution_pb2.ResponsePingMessage(**result)
        return response

    def SubmitTask(self, task, context):
        # Process the task
        print(f"Submit task to task db")
        db_path = get_db_path(config.workspace)
        print(db_path)
        result = submit_task_to_db(task=task, task_submitter=task_submitter, db_path=get_db_path(config.workspace))
        # Return a response
        response = neural_solution_pb2.TaskResponse(**result)
        return response

    def GetTaskById(self, task_id, context):
        db_path = get_db_path(config.workspace)
        result = query_task_status(task_id.task_id, db_path)
        print(f"query result : result")
        response = neural_solution_pb2.TaskStatus(**result)
        return response

    def QueryTaskResult(self, task_id, context):
        db_path = get_db_path(config.workspace)
        result = query_task_result(task_id.task_id, db_path, config.workspace)
        result['tuning_information'] = dict_to_str(result["tuning_information"])
        result['optimization_result'] = dict_to_str(result["optimization_result"])
        response = neural_solution_pb2.ResponseTaskResult(**result)
        return response


def serve():
    port = str(config.grpc_api_port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    neural_solution_pb2_grpc.add_TaskServiceServicer_to_server(
        TaskSubmitterServicer(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Frontend with RESTful API")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", \
        help="The address to submit task.")
    parser.add_argument("-FP", "--grpc_api_port", type=int, default=8001, \
        help="Port to submit task by user.")
    parser.add_argument("-TMP", "--task_monitor_port", type=int, default=2222, \
        help="Port to monitor task.")
    parser.add_argument("-RMP", "--result_monitor_port", type=int, default=3333, \
        help="Port to monitor result.")
    parser.add_argument("-WS", "--workspace", type=str, default="./ns_workspace", \
        help="Work space.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig()
    args = parse_arguments()
    print(args.workspace)
    config.workspace = args.workspace
    config.grpc_api_port = config.grpc_api_port
    serve()

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

import argparse
import logging
from concurrent import futures

import grpc

from neural_solution.config import config
from neural_solution.frontend.gRPC.proto import neural_solution_pb2, neural_solution_pb2_grpc
from neural_solution.frontend.task_submitter import task_submitter
from neural_solution.frontend.utility import (
    check_service_status,
    query_task_result,
    query_task_status,
    submit_task_to_db,
)
from neural_solution.utils import logger
from neural_solution.utils.utility import dict_to_str, get_db_path


class TaskSubmitterServicer(neural_solution_pb2_grpc.TaskServiceServicer):
    """Deliver services.

    Args:
        neural_solution_pb2_grpc (): task servicer
    """

    def __init__(self) -> None:
        """Init."""
        pass

    def Ping(self, empty_msg, context):
        """Check service status.

        Args:
            empty_msg (str): empty message
            context (str): context

        Returns:
            Response: service status
        """
        print(f"Ping grpc serve.")
        port_lst = [config.result_monitor_port]
        result = check_service_status(port_lst, service_address=config.service_address)
        response = neural_solution_pb2.ResponsePingMessage(**result)  # pylint: disable=no-member
        return response

    def SubmitTask(self, task, context):
        """Submit task.

        Args:
            task (Task): task object
            Fields:
                task_id: The task id
                arguments: The task command
                workers: The requested resource unit number
                status: The status of the task: pending/running/done
                result: The result of the task, which is only value-assigned when the task is done

        Returns:
            json: status , id of task and messages.
        """
        # Process the task
        print(f"Submit task to task db")
        db_path = get_db_path(config.workspace)
        print(db_path)
        result = submit_task_to_db(task=task, task_submitter=task_submitter, db_path=get_db_path(config.workspace))
        # Return a response
        response = neural_solution_pb2.TaskResponse(**result)  # pylint: disable=no-member
        return response

    def GetTaskById(self, task_id, context):
        """Get task status, result, quantized model path according to id.

        Args:
            task_id (str): the id of task.

        Returns:
            json: task status, result, quantized model path
        """
        db_path = get_db_path(config.workspace)
        result = query_task_status(task_id.task_id, db_path)
        print(f"query result : result")
        response = neural_solution_pb2.TaskStatus(**result)  # pylint: disable=no-member
        return response

    def QueryTaskResult(self, task_id, context):
        """Get task status and information according to id.

        Args:
            task_id (str): the id of task.

        Returns:
            json: task status and information
        """
        db_path = get_db_path(config.workspace)
        result = query_task_result(task_id.task_id, db_path, config.workspace)
        result["tuning_information"] = dict_to_str(result["tuning_information"])
        result["optimization_result"] = dict_to_str(result["optimization_result"])
        response = neural_solution_pb2.ResponseTaskResult(**result)  # pylint: disable=no-member
        return response


def serve():
    """Service entrance."""
    port = str(config.grpc_api_port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    neural_solution_pb2_grpc.add_TaskServiceServicer_to_server(TaskSubmitterServicer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def parse_arguments():
    """Parse the command line options."""
    parser = argparse.ArgumentParser(description="Frontend with gRPC API")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="The address to submit task.")
    parser.add_argument("-FP", "--grpc_api_port", type=int, default=8001, help="Port to submit task by user.")
    parser.add_argument("-TMP", "--task_monitor_port", type=int, default=2222, help="Port to monitor task.")
    parser.add_argument("-RMP", "--result_monitor_port", type=int, default=3333, help="Port to monitor result.")
    parser.add_argument("-WS", "--workspace", type=str, default="./ns_workspace", help="Work space.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info(f"Try to start gRPC server.")
    logging.basicConfig()
    args = parse_arguments()
    print(args.workspace)
    config.workspace = args.workspace
    config.grpc_api_port = args.grpc_api_port
    serve()

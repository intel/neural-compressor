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

import argparse
import json
import os

import grpc

from neural_solution.config import config
from neural_solution.frontend.gRPC.proto import neural_solution_pb2, neural_solution_pb2_grpc
from neural_solution.utils import logger


def _parse_task_from_json(request_path):
    file_path = os.path.abspath(request_path)
    with open(file_path) as fp:
        task = json.load(fp)
    return task


def submit_task(args):
    """Implement main entry point for the client of gRPC frontend."""
    task = _parse_task_from_json(args.request)
    logger.info("Parsed task:")
    logger.info(task)

    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel("localhost:" + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    # Ping serve
    request = neural_solution_pb2.EmptyRequest()  # pylint: disable=no-member
    response = stub.Ping(request)
    logger.info(response.status)
    logger.info(response.msg)

    # Create a task request with the desired fields
    request = neural_solution_pb2.Task(  # pylint: disable=no-member
        script_url=task["script_url"],
        optimized=task["optimized"] == "True",
        arguments=task["arguments"],
        approach=task["approach"],
        requirements=task["requirements"],
        workers=task["workers"],
    )

    # Call the SubmitTask RPC on the server
    response = stub.SubmitTask(request)

    # Process the response
    logger.info(response.status)
    logger.info(response.task_id)
    logger.info(response.msg)


def run_query_task_result(args):
    """Query task result according to id.

    Args:
        args: args includes task_id
    """
    task_id = args.task_id
    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel("localhost:" + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    request = neural_solution_pb2.TaskId(task_id=task_id)  # pylint: disable=no-member
    response = stub.QueryTaskResult(request)
    logger.info(response.status)
    logger.info(response.tuning_information)
    logger.info(response.optimization_result)


def run_query_task_status(args):
    """Query task status according to id.

    Args:
        args: args includes task_id
    """
    task_id = args.task_id
    # Create a gRPC channel
    port = str(config.grpc_api_port)
    channel = grpc.insecure_channel("localhost:" + port)

    # Create a stub (client)
    stub = neural_solution_pb2_grpc.TaskServiceStub(channel)

    request = neural_solution_pb2.TaskId(task_id=task_id)  # pylint: disable=no-member
    response = stub.GetTaskById(request)
    logger.info(response.status)
    logger.info(response.optimized_result)
    logger.info(response.result_path)


if __name__ == "__main__":
    logger.info("Try to start gRPC server.")
    """Parse the command line options."""
    parser = argparse.ArgumentParser(description="gRPC Client")
    subparsers = parser.add_subparsers(help="Action", dest="action")

    submit_action_parser = subparsers.add_parser("submit", help="Submit help")

    submit_action_parser.set_defaults(func=submit_task)
    submit_action_parser.add_argument("--request", type=str, default=None, help="Request json file path.")

    query_action_parser = subparsers.add_parser("query", help="Query help")
    query_action_parser.set_defaults(func=run_query_task_result)
    query_action_parser.add_argument("--task_id", type=str, default=None, help="Query task by task id.")

    parser.add_argument("--grpc_api_port", type=str, default="8001", help="grpc server port.")
    parser.add_argument("--result_monitor_port", type=str, default="2222", help="result monitor port.")
    parser.add_argument("--task_monitor_port", type=str, default="3333", help="task monitor port.")

    args = parser.parse_args()
    config.grpc_api_port = args.grpc_api_port
    config.result_monitor_port = args.result_monitor_port
    config.task_monitor_port = args.task_monitor_port
    args.func(args)

# for test:
# python client.py query --task_id="d3e10a49326449fb9d0d62f2bfc1cb43"
# python client.py submit --request="test_task_request.json"

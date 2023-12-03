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
"""Neural Solution task submitter."""

import json
import socket

from pydantic import BaseModel  # pylint: disable=no-name-in-module

from neural_solution.config import config


class Task(BaseModel):
    """Task definition for submitting requests.

    Args:
        BaseModel (ModelMetaclass): meta class
    """

    script_url: str
    optimized: bool
    arguments: list
    approach: str
    requirements: list
    workers: int


class TaskSubmitter:
    """Responsible for submitting tasks."""

    def __init__(self, task_monitor_port=2222, result_monitor_port=3333, service_address="localhost"):
        """Init.

        Args:
            task_monitor_port (int, optional): the port for monitoring tasks. Defaults to 2222.
            result_monitor_port (int, optional): the port for monitoring results. Defaults to 3333.
            service_address (str, optional): the service address. Defaults to "localhost".
        """
        self.task_monitor_port = task_monitor_port
        self.result_monitor_port = result_monitor_port
        self.service_address = service_address

    def serialize(self, tid: str) -> bytes:
        """Serialize a str object."""
        d = {"task_id": tid}
        return json.dumps(d).encode()

    def submit_task(self, tid):
        """Submit task by sending id.

        Args:
            tid (str): the id of task
        """
        s = socket.socket()
        s.connect((self.service_address, self.task_monitor_port))
        s.send(self.serialize(tid))
        s.close()


task_submitter = TaskSubmitter(
    task_monitor_port=config.task_monitor_port, result_monitor_port=config.result_monitor_port
)

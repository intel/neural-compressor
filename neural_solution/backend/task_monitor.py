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
"""Neural Solution task monitor."""
import socket

from neural_solution.backend.utils.utility import deserialize, serialize
from neural_solution.utils import logger


class TaskMonitor:
    """TaskMonitor is a thread that monitors the coming tasks and appends them to the task queue.

    Attributes:
        port: the port that the task monitor listens to
        task_db: the TaskDb object that manages the tasks
    """

    def __init__(self, port, task_db):
        """Init TaskMonitor."""
        self.s = socket.socket()
        self.port = port
        self.task_db = task_db

    def _start_listening(self, host, port, max_parallelism):
        self.s.bind(("localhost", port))  # open a port as the serving port for tasks
        self.s.listen(max_parallelism)

    def _receive_task(self):
        c, addr = self.s.accept()
        task = c.recv(4096)
        task_dict = deserialize(task)
        if "ping" in task_dict:
            logger.info("[TaskMonitor] Client query status.")
            c.send(b"ok")
            return False
        task_id = task_dict["task_id"]

        logger.info("[TaskMonitor] getting task: {}".format(task_id))
        return self.task_db.get_task_by_id(task_id)
        # return Task(task_id=task["task_id"], arguments=task["arguments"],
        #     workers=task["workers"], status="pending", script_url=task['script_url'])

    def _append_task(self, task):
        self.task_db.append_task(task)
        logger.info("[TaskMonitor] append task {} done.".format(task.task_id))

    def wait_new_task(self):
        """Monitor the coming tasks and append it to the task db."""
        self._start_listening("localhost", self.port, 10)
        while True:
            logger.info("[TaskMonitor] waiting for new tasks...")
            task = self._receive_task()
            if not task:
                continue
            self._append_task(task)

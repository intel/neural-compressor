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
"""Neural Solution result monitor."""

import socket

from neural_solution.backend.task_db import TaskDB
from neural_solution.backend.utils.utility import deserialize, serialize
from neural_solution.utils import logger


class ResultMonitor:
    """ResultMonitor is a thread that monitors the coming task results and update the task collection in the TaskDb.

    Attributes:
        port: The port that ResultMonitor listens to
        task_db: the TaskDb object that manages the tasks
    """

    def __init__(self, port, task_db: TaskDB):
        """Init ResultMonitor.

        Args:
            port (int): the port for monitoring task results.
            task_db (TaskDB): the object of TaskDB.
        """
        self.s = socket.socket()
        self.port = port
        self.task_db = task_db

    def wait_result(self):
        """Monitor the task results and update them in the task db and send back to studio."""
        self.s.bind(("localhost", self.port))  # open a port as the serving port for results
        self.s.listen(10)
        while True:
            logger.info("[ResultMonitor] waiting for results...")
            c, addr = self.s.accept()
            result = c.recv(2048)
            result = deserialize(result)
            if "ping" in result:
                logger.info("[ResultMonitor] Client query status.")
                c.send(b"ok")
                c.close()
                continue
            logger.info("[ResultMonitor] getting result: {}".format(result))
            logger.info("[ResultMonitor] getting q_model path: {}".format(result["q_model_path"]))
            self.task_db.update_q_model_path_and_result(result["task_id"], result["q_model_path"], result["result"])
            c.close()
            # TODO send back the result to the studio
            # or let studio manually fresh the page and call the query_task_status to get the result?

    def query_task_status(self, task_id):
        """Synchronize query on the task status."""
        # TODO send back the result to the studio? RPC for query?
        logger.info(self.task_db.lookup_task_status(task_id))

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
"""Neural Solution backend utils."""
import json
import os
from urllib.parse import urlparse

from neural_solution.utils import logger


def serialize(request: dict) -> bytes:
    """Serialize a dict object to bytes for inter-process communication."""
    return json.dumps(request).encode()


def deserialize(request: bytes) -> dict:
    """Deserialize the recived bytes to a dict object."""
    return json.loads(request)


def dump_elapsed_time(customized_msg=""):
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """
    import time

    def f(func):
        def fi(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            logger.info(
                "%s elapsed time: %s ms"
                % (customized_msg if customized_msg else func.__qualname__, round((end - start) * 1000, 2))
            )
            return res

        return fi

    return f


def get_task_log_path(log_path, task_id):
    """Get the path of task log according id.

    Args:
        log_path (str): the log path of task
        task_id (str): the task id

    Returns:
        str: the path of task log file
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_path = "{}/task_{}.txt".format(log_path, task_id)
    return log_file_path


def get_db_path(workspace="./"):
    """Get the database path.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the path of database
    """
    return os.path.join(workspace, "db", "task.db")


def get_task_workspace(workspace="./"):
    """Get the workspace of task.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the workspace of task
    """
    return os.path.join(workspace, "task_workspace")


def get_task_log_workspace(workspace="./"):
    """Get the log workspace for task.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: the log workspace for task
    """
    return os.path.join(workspace, "task_log")


def get_serve_log_workspace(workspace="./"):
    """Get log workspace for service.

    Args:
        workspace (str, optional): the workspace for Neural Solution. Defaults to "./".

    Returns:
        str: log workspace for service
    """
    return os.path.join(workspace, "serve_log")


def build_local_cluster(db_path):
    """Build a local cluster.

    Args:
        db_path (str): database path

    Returns:
        (Cluster, int): cluster and num threads per process
    """
    from neural_solution.backend.cluster import Cluster, Node

    hostname = "localhost"
    node1 = Node(name=hostname, num_sockets=2, num_cores_per_socket=5)
    node2 = Node(name=hostname, num_sockets=2, num_cores_per_socket=5)
    node3 = Node(name=hostname, num_sockets=2, num_cores_per_socket=5)

    node_lst = [node1, node2, node3]
    cluster = Cluster(node_lst=node_lst, db_path=db_path)
    return cluster, 5


def build_cluster(file_path, db_path):
    """Build cluster according to the host file.

    Args:
        file_path :  the path of host file.

    Returns:
        Cluster: return cluster object.
    """
    from neural_solution.backend.cluster import Cluster, Node

    # If no file is specified, build a local cluster
    if file_path == "None" or file_path is None:
        return build_local_cluster(db_path)

    if not os.path.exists(file_path):
        raise Exception(f"Please check the path of host file: {file_path}.")

    node_lst = []
    num_threads_per_process = 5
    with open(file_path, "r") as f:
        for line in f:
            hostname, num_sockets, num_cores_per_socket = line.strip().split(" ")
            num_sockets, num_cores_per_socket = int(num_sockets), int(num_cores_per_socket)
            node = Node(name=hostname, num_sockets=num_sockets, num_cores_per_socket=num_cores_per_socket)
            node_lst.append(node)
            num_threads_per_process = num_cores_per_socket
    cluster = Cluster(node_lst=node_lst, db_path=db_path)
    return cluster, num_threads_per_process


def get_current_time():
    """Get current time.

    Returns:
        str: the current time in hours, minutes, and seconds.
    """
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def synchronized(func):
    """Locking for synchronization.

    Args:
        func (function): decorative function
    """

    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


def build_workspace(path, task_id=""):
    """Build workspace of running tasks.

    Args:
        path: master work directory for all tasks.
        task_id: the id of task
    """
    task_path = "{}/{}".format(path, task_id)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    return os.path.abspath(task_path)


def is_remote_url(url_or_filename):
    """Check if input is a URL.

    Args:
        url_or_filename (str): url_or_filename

    Returns:
        bool: True or False
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def create_dir(path):
    """Create the (nested) path if not exist."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def get_q_model_path(log_path):
    """Get the quantized model path from task log.

    Args:
        log_path (str): log path for task

    Returns:
        str: quantized model path
    """
    import re

    for line in reversed(open(log_path).readlines()):
        match = re.search(r"(Save quantized model to|Save config file and weights of quantized model to) (.+?)\.", line)
        if match:
            q_model_path = match.group(2)
            return q_model_path
    return "quantized model path not found"

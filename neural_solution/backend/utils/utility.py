import json
import os
from ..utils import logger
from urllib.parse import urlparse
from ..constant import NUM_CORES_PER_SOCKET, NUM_SOCKETS, TASK_WORKSPACE, TASK_LOG_path


def serialize(request: dict) -> bytes:
    """Serialize a dict object to bytes for inter-process communication."""
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    """Deserialize the recived bytes to a dict object"""
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
            logger.info('%s elapsed time: %s ms' %(customized_msg if customized_msg else func.__qualname__,
                                             round((end - start) * 1000, 2)))
            return res
        return fi
    return f

def get_task_log_path(task_id):
    """Get the task log file path.

    Args:
        task_id: _description_
    """
    if not os.path.exists(TASK_LOG_path):
        os.makedirs(TASK_LOG_path)
    log_file_path =  "{}/task_{}.txt".format(TASK_LOG_path,task_id)
    return log_file_path

def build_local_cluster():
    from backend.cluster import Node, Cluster
    hostname = 'localhost'
    node1 = Node(name=hostname,num_sockets=2, num_cores_per_socket=5)
    node2 = Node(name=hostname,num_sockets=2, num_cores_per_socket=5)
    node3 = Node(name=hostname,num_sockets=2, num_cores_per_socket=5)

    node_lst = [node1, node2, node3]
    cluster = Cluster(node_lst=node_lst)
    return cluster

def build_cluster(file_path):
    """Build cluster according to the host file.

    Args:
        file_path :  the path of host file.

    Returns:
        Cluster: return cluster object.
    """
    from backend.cluster import Node, Cluster
    # If no file is specified, build a local cluster
    if file_path == "None" or file_path is None:
        return build_local_cluster()

    if not os.path.exists(file_path):
        logger.info(f"Please check the path of host file: {file_path}.")
        exit(1)

    node_lst = []
    with open(file_path, 'r') as f:
        for line in f:
            hostname = line.strip()
            node = Node(name=hostname, num_sockets=NUM_SOCKETS, num_cores_per_socket=NUM_CORES_PER_SOCKET)
            node_lst.append(node)
    cluster = Cluster(node_lst=node_lst)
    return cluster

def get_current_time():
    from datetime import datetime
    return datetime.now().strftime('%H:%M:%S')

def synchronized(func):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)
    return wrapper

def build_workspace(path=TASK_WORKSPACE, task_id=""):
    """Build workspace of running tasks.

    Args:
        path: master work directory for all tasks.
        task_id: _description_
    """
    task_path = "{}/{}".format(path, task_id)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    return os.path.abspath(task_path)

def is_remote_url(url_or_filename):
    """Check if input is a URL

    Args:
        url_or_filename (str): url_or_filename

    Returns:
        bool: True or False
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def create_dir(path):
    """Create the (nested) path if not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
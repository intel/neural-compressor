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

import threading
import os
import sqlite3
from typing import List
from .task import Task
from .utils.utility import synchronized, create_dir
from .utils import logger
from .constant import DB_PATH
from collections import Counter



class Cluster:
    def __init__(self, node_lst=[]):
        """Cluster manages all the server resources.

        TaskDb provides atomic operations on managing the resource dict.

        Attributes:
            resource_dict: a mapping from "<ip>-<socket index>" to 0/1 (idle/occupied)
                e.g "localhost-1":0 means socket 1 of localhost is idle
                and "192.168.x.x-0":1 means socket 0 of 192.168.x.x is in use
            lock: the lock on the data structures to provide atomic operations
        """
        # TODO Replace resource_dict with node_lst
        # self.resource_dict = resource_dict  # {"localhost-0":0, "localhost-1":1, "192.168.x.x-0":1}
        self.lock = threading.Lock()
        self.node_lst = node_lst
        self.socket_queue = []
        create_dir(DB_PATH)
        self.conn = sqlite3.connect(f'{DB_PATH}/task.db', check_same_thread=False)
        self.initial_cluster_from_node_lst(node_lst)
        self.lock = threading.Lock()

    def reserve_resource(self, task):
        """Reserve the resource and return the requested list of resources."""
        reserved_resource_lst = []
        workers = task.workers
        logger.info(f"task {task.task_id} needs {workers}")
        reserved_resource_lst = self.get_free_socket(workers)
        if reserved_resource_lst:
            allocated_resources = {}
            counts = Counter(int(item.split()[0]) for item in reserved_resource_lst)

            for node_id, count in counts.items():
                allocated_resources[node_id] = count
            for node_id in allocated_resources:
                sql = """
                        UPDATE cluster
                        SET busy_sockets = busy_sockets + ?,
                            free_sockets = total_sockets - busy_sockets - ?
                        WHERE id = ?
                    """
                self.cursor.execute(sql, (allocated_resources[node_id], allocated_resources[node_id], node_id))
            self.conn.commit()
            logger.info(f"[Cluster] Assign {reserved_resource_lst} to task {task.task_id}")
        return reserved_resource_lst


    @synchronized
    def free_resource(self, reserved_resource_lst):
        """Free the resource by adding the previous occupied resources to the socket queue."""
        self.socket_queue += reserved_resource_lst
        counts = Counter(int(item.split()[0]) for item in reserved_resource_lst)
        free_resources = {}
        for node_id, count in counts.items():
                free_resources[node_id] = count
        for node_id, count in counts.items():
                free_resources[node_id] = count
        for node_id in free_resources:
            sql = """
                    UPDATE cluster
                    SET free_sockets = free_sockets + ?,
                        busy_sockets = total_sockets - free_sockets - ?
                    WHERE id = ?
                """
            self.cursor.execute(sql, (free_resources[node_id], free_resources[node_id], node_id))
            self.conn.commit()
        logger.info(f"[Cluster] free resource {reserved_resource_lst}, now have free resource {self.socket_queue}")

    @synchronized
    def get_free_socket(self, num_sockets: int) -> List[str]:
        """Get the free sockets list."""
        booked_socket_lst = []
        if len(self.socket_queue) < num_sockets:
            logger.info(f"Can not allocate {num_sockets} sockets, due to only {len(self.socket_queue)} left.")
            return 0
        else:
            booked_socket_lst = self.socket_queue[:num_sockets]
            self.socket_queue = self.socket_queue[num_sockets:]
        return booked_socket_lst

    @synchronized
    def initial_cluster_from_node_lst(self, node_lst):
        self.conn = sqlite3.connect(f'{DB_PATH}/task.db', check_same_thread=False)  # sqlite should set this check_same_thread to False
        self.cursor = self.conn.cursor()
        self.cursor.execute('drop table if exists cluster ')
        self.cursor.execute(r'create table cluster(id INTEGER PRIMARY KEY AUTOINCREMENT,' +
             'node_info varchar(500),' +
             'status varchar(100),' +
             'free_sockets int,' +
             'busy_sockets int,' +
             'total_sockets int)')
        self.node_lst = node_lst
        for index, node in enumerate(self.node_lst):
            self.socket_queue += [str(index+1) + " " + node.name] * node.num_sockets
            self.cursor.execute(r"insert into cluster(node_info, status, free_sockets, busy_sockets, total_sockets)" +
                    "values ('{}', '{}', {}, {}, {})".format(repr(node).replace("Node", f"Node{index+1}"),
                                                             "alive",
                                                             node.num_sockets,
                                                             0,
                                                             node.num_sockets))

        self.conn.commit()
        logger.info(f"socket_queue:  {self.socket_queue}")

class Node:
    name: str = "unknown_node"
    ip: str = "unknown_ip"
    num_sockets: int = 0
    num_cores_per_socket: int = 0
    num_gpus: int = 0 # For future use

    def __init__(self,
                 name: str,
                 ip: str = "unknown_ip",
                 num_sockets: int = 0,
                 num_cores_per_socket: int = 0,
                 num_gpus: int = 0) -> None:
        """Init node.

        Args:
            name: node name
            ip: ip address. Defaults to "unknown_ip".
            num_sockets: the number of sockets. Defaults to 0.
            num_cores_per_socket: the number of core(s) per socket. Defaults to 0.
            num_gpus: the number of gpus. Defaults to 0.
        """
        self.name = name
        self.ip = ip
        self.num_sockets = num_sockets
        self.num_cores_per_socket = num_cores_per_socket
        self.num_gpus = num_gpus

    def __repr__(self) -> str:
        return f"Node: {self.name}(ip: {self.ip}) has {self.num_sockets} socket(s) " \
            f"and each socket has {self.num_cores_per_socket} cores."




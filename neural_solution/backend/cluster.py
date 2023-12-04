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
"""Neural Solution cluster."""
import sqlite3
import threading
from collections import Counter
from typing import List

from neural_solution.backend.utils.utility import create_dir, synchronized
from neural_solution.utils import logger


class Cluster:
    """Cluster resource management based on sockets."""

    def __init__(self, node_lst=[], db_path=None):
        """Init Cluster.

        Args:
            node_lst: node list. Defaults to [].
            db_path: cluster db path. Defaults to None.
        """
        self.lock = threading.Lock()
        self.node_lst = node_lst
        self.socket_queue = []
        self.db_path = db_path
        create_dir(db_path)
        self.conn = sqlite3.connect(f"{db_path}", check_same_thread=False)
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
        # delete nodes with status of remove, some version without RETURNING syntax
        self.cursor.execute("SELECT id FROM cluster WHERE status='remove' AND busy_sockets=0")
        deleted_ids = self.cursor.fetchall()
        deleted_ids = [str(id_tuple[0]) for id_tuple in deleted_ids]
        self.cursor.execute("DELETE FROM cluster WHERE status='remove' AND busy_sockets=0")
        self.conn.commit()

        # remove deleted nodes from socket queue
        socket_queue_delete_ids = [socket for socket in self.socket_queue if socket.split()[0] in deleted_ids]
        if len(socket_queue_delete_ids) > 0:
            logger.info(f"[Cluster] remove node-list {socket_queue_delete_ids} from socket_queue:  {self.socket_queue}")
            self.socket_queue = [socket for socket in self.socket_queue if socket.split()[0] not in deleted_ids]
        logger.info(f"[Cluster] free resource {reserved_resource_lst}, now have free resource {self.socket_queue}")

    @synchronized
    def get_free_socket(self, num_sockets: int) -> List[str]:
        """Get the free sockets list."""
        booked_socket_lst = []

        # detect and append new resource
        self.cursor.execute("SELECT id, name, total_sockets FROM cluster where status = 'join'")
        new_node_lst = self.cursor.fetchall()
        for index, name, total_sockets in new_node_lst:
            sql = """
                    UPDATE cluster
                    SET status = ?
                    WHERE id = ?
                """
            self.cursor.execute(sql, ("alive", index))
            self.conn.commit()
            self.socket_queue += [str(index) + " " + name] * total_sockets
            logger.info(f"[Cluster] add new node-id {index} to socket_queue:  {self.socket_queue}")

        # do not assign nodes with status of remove
        # remove to-delete nodes from socket queue
        self.cursor.execute("SELECT id FROM cluster WHERE status='remove'")
        deleted_ids = self.cursor.fetchall()
        deleted_ids = [str(id_tuple[0]) for id_tuple in deleted_ids]

        socket_queue_delete_ids = [socket for socket in self.socket_queue if socket.split()[0] in deleted_ids]
        if len(socket_queue_delete_ids) > 0:
            logger.info(f"[Cluster] remove node-list {socket_queue_delete_ids} from socket_queue:  {self.socket_queue}")
            self.socket_queue = [socket for socket in self.socket_queue if socket.split()[0] not in deleted_ids]

        # delete nodes with status of remove
        self.cursor.execute("DELETE FROM cluster WHERE status='remove' AND busy_sockets=0")
        self.conn.commit()

        if len(self.socket_queue) < num_sockets:
            logger.info(f"Can not allocate {num_sockets} sockets, due to only {len(self.socket_queue)} left.")
            return 0
        else:
            booked_socket_lst = self.socket_queue[:num_sockets]
            self.socket_queue = self.socket_queue[num_sockets:]
        return booked_socket_lst

    @synchronized
    def initial_cluster_from_node_lst(self, node_lst):
        """Initialize cluster according to the node list.

        Args:
            node_lst (List): the node list.
        """
        # sqlite should set this check_same_thread to False
        self.conn = sqlite3.connect(f"{self.db_path}", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("drop table if exists cluster ")
        self.cursor.execute(
            r"create table cluster(id INTEGER PRIMARY KEY AUTOINCREMENT,"
            + "name varchar(100),"
            + "node_info varchar(500),"
            + "status varchar(100),"
            + "free_sockets int,"
            + "busy_sockets int,"
            + "total_sockets int)"
        )
        self.node_lst = node_lst
        for index, node in enumerate(self.node_lst):
            self.socket_queue += [str(index + 1) + " " + node.name] * node.num_sockets
            self.cursor.execute(
                r"insert into cluster(name, node_info, status, free_sockets, busy_sockets, total_sockets)"
                + "values ('{}', '{}', '{}', {}, {}, {})".format(
                    node.name,
                    repr(node).replace("Node", f"Node{index+1}"),
                    "alive",
                    node.num_sockets,
                    0,
                    node.num_sockets,
                )
            )

        self.conn.commit()
        logger.info(f"socket_queue:  {self.socket_queue}")


class Node:
    """Node definition."""

    name: str = "unknown_node"
    ip: str = "unknown_ip"
    num_sockets: int = 0
    num_cores_per_socket: int = 0
    num_gpus: int = 0  # For future use

    def __init__(
        self, name: str, ip: str = "unknown_ip", num_sockets: int = 0, num_cores_per_socket: int = 0, num_gpus: int = 0
    ) -> None:
        """Init node.

        hostfile template:
        host1 2 20 # host1 has 2 sockets, each socket has 20 cores
        host2 2 20 # host2 has 2 sockets, each socket has 20 cores

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
        """Return node info.

        Returns:
            str: node info.
        """
        return (
            f"Node: {self.name}(ip: {self.ip}) has {self.num_sockets} socket(s) "
            f"and each socket has {self.num_cores_per_socket} cores."
        )

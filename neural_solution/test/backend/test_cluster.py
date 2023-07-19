"""Tests for cluster"""
import importlib
import os
import shutil
import unittest

from neural_solution.backend.cluster import Cluster, Node
from neural_solution.backend.task import Task
from neural_solution.utils.utility import get_db_path, get_task_log_workspace, get_task_workspace

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
db_path = get_db_path(NEURAL_SOLUTION_WORKSPACE)


class TestCluster(unittest.TestCase):
    @classmethod
    def setUp(self):
        node_lst = [Node("node1", "localhost", 2, 4), Node("node2", "localhost", 2, 4)]
        self.cluster = Cluster(node_lst, db_path=db_path)

        self.task = Task(
            task_id="1",
            arguments=["arg1", "arg2"],
            workers=2,
            status="pending",
            script_url="https://example.com",
            optimized=True,
            approach="static",
            requirement=["req1", "req2"],
            result="",
            q_model_path="q_model_path",
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("ns_workspace")

    def test_reserve_resource(self):
        task = self.task
        reserved_resource_lst = self.cluster.reserve_resource(task)
        self.assertEqual(len(reserved_resource_lst), 2)
        self.assertEqual(self.cluster.socket_queue, ["2 node2", "2 node2"])

    def test_free_resource(self):
        task = self.task
        reserved_resource_lst = self.cluster.reserve_resource(task)
        self.cluster.free_resource(reserved_resource_lst)
        self.assertEqual(self.cluster.socket_queue, ["2 node2", "2 node2", "1 node1", "1 node1"])

    def test_get_free_socket(self):
        free_socket_lst = self.cluster.get_free_socket(4)
        self.assertEqual(len(free_socket_lst), 4)
        self.assertEqual(free_socket_lst, ["1 node1", "1 node1", "2 node2", "2 node2"])
        self.assertEqual(self.cluster.socket_queue, [])

        # Attempting to over allocate resources
        free_socket_lst = self.cluster.get_free_socket(10)
        self.assertEqual(free_socket_lst, 0)


if __name__ == "__main__":
    unittest.main()

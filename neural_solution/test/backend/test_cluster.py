"""Tests for cluster"""
import importlib
import shutil
import os
import unittest

from neural_solution.backend.cluster import Node, Cluster

class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        ns_path = os.path.dirname(importlib.util.find_spec('neural_solution').origin)
        self.workspace = os.path.join(ns_path, '../', 'ns_workspace')
        print(self.workspace)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.workspace)

    def test_cluster(self):
        node1 = Node(name='node1',num_sockets=2, num_cores_per_socket=20)
        node2 = Node(name='node2',num_sockets=2, num_cores_per_socket=20)
        node3 = Node(name='node3',num_sockets=2, num_cores_per_socket=20)
        node_lst = [node1, node2, node3]
        cluster = Cluster(node_lst=node_lst)

if __name__ == "__main__":
    unittest.main()

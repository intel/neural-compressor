import os
import shutil
import unittest
from unittest.mock import MagicMock, mock_open, patch

from neural_solution.backend.utils.utility import (
    build_cluster,
    build_local_cluster,
    build_workspace,
    create_dir,
    deserialize,
    dump_elapsed_time,
    get_current_time,
    get_q_model_path,
    get_task_log_path,
    is_remote_url,
    serialize,
    synchronized,
)
from neural_solution.config import config
from neural_solution.utils.utility import get_db_path, get_task_log_workspace, get_task_workspace

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
DB_PATH = NEURAL_SOLUTION_WORKSPACE + "/db"
TASK_WORKSPACE = NEURAL_SOLUTION_WORKSPACE + "/task_workspace"
TASK_LOG_path = NEURAL_SOLUTION_WORKSPACE + "/task_log"
SERVE_LOG_PATH = NEURAL_SOLUTION_WORKSPACE + "/serve_log"

config.workspace = NEURAL_SOLUTION_WORKSPACE
db_path = get_db_path(config.workspace)


class TestUtils(unittest.TestCase):
    @classmethod
    def tearDown(self) -> None:
        if os.path.exists("ns_workspace"):
            shutil.rmtree("ns_workspace")

    def test_serialize(self):
        input_dict = {"key1": "value1", "key2": "value2"}
        expected_output = b'{"key1": "value1", "key2": "value2"}'
        self.assertEqual(serialize(input_dict), expected_output)

    def test_deserialize(self):
        input_bytes = b'{"key1": "value1", "key2": "value2"}'
        expected_output = {"key1": "value1", "key2": "value2"}
        self.assertEqual(deserialize(input_bytes), expected_output)

    def test_dump_elapsed_time(self):
        @dump_elapsed_time("test function")
        def test_function():
            return True

        with patch("neural_solution.utils.logger") as mock_logger:
            test_function()

    def test_get_task_log_path(self):
        task_id = 123
        expected_output = f"{TASK_LOG_path}/task_{task_id}.txt"
        self.assertEqual(
            get_task_log_path(log_path=get_task_log_workspace(config.workspace), task_id=task_id), expected_output
        )

    def test_build_local_cluster(self):
        with patch("neural_solution.backend.cluster.Node") as mock_node, patch(
            "neural_solution.backend.cluster.Cluster"
        ) as mock_cluster:
            mock_node_obj = MagicMock()
            mock_node.return_value = mock_node_obj
            mock_node_obj.name = "localhost"
            mock_node_obj.num_sockets = 2
            mock_node_obj.num_cores_per_socket = 5
            build_local_cluster(db_path=db_path)
            mock_node.assert_called_with(name="localhost", num_sockets=2, num_cores_per_socket=5)
            mock_cluster.assert_called_once()

    def test_build_cluster(self):
        # Test 2 hostname
        path = "test.txt"
        with open(path, "w") as f:
            f.write("hostname1 2 20\nhostname2 2 20")
        cluster, _ = build_cluster(path, db_path=db_path)
        self.assertIsNotNone(cluster)

        os.remove("test.txt")

        file_path = "test_host_file"
        with patch("neural_solution.backend.cluster.Node") as mock_node, patch(
            "neural_solution.backend.cluster.Cluster"
        ) as mock_cluster, patch("builtins.open") as mock_open, patch("os.path.exists") as mock_exists:
            # Test None
            cluster, _ = build_cluster(file_path=None, db_path=db_path)
            mock_cluster.assert_called()

            mock_exists.return_value = True
            build_cluster(file_path, db_path=db_path)

        # test_build_cluster_file_not_exist
        file_path = "test_file"
        with patch("neural_solution.backend.cluster.Node"), patch("neural_solution.backend.cluster.Cluster"), patch(
            "builtins.open"
        ), patch("os.path.exists") as mock_exists, patch("neural_solution.utils.logger") as mock_logger:
            mock_exists.return_value = False
            self.assertRaises(Exception, build_cluster, file_path)
            mock_logger.reset_mock()

    def test_get_current_time(self):
        self.assertIsNotNone(get_current_time())

    def test_synchronized(self):
        class TestClass:
            def __init__(self):
                self.lock = MagicMock()

            @synchronized
            def test_function(self):
                return True

        test_class = TestClass()
        with patch.object(test_class, "lock"):
            test_class.test_function()

    def test_build_workspace(self):
        task_id = 123
        expected_output = os.path.abspath(f"{TASK_WORKSPACE}/{task_id}")
        self.assertEqual(build_workspace(path=get_task_workspace(config.workspace), task_id=task_id), expected_output)

    def test_is_remote_url(self):
        self.assertTrue(is_remote_url("http://test.com"))
        self.assertTrue(is_remote_url("https://test.com"))
        self.assertFalse(is_remote_url("test.txt"))

    def test_create_dir(self):
        path = "test/path/test.txt"
        create_dir(path)
        self.assertTrue(os.path.exists(os.path.dirname(path)))

    @patch("builtins.open", mock_open(read_data="Save quantized model to /path/to/model."))
    def test_get_q_model_path_success(self):
        log_path = "fake_log_path"
        q_model_path = get_q_model_path(log_path, "task_id")
        self.assertEqual(q_model_path, "/path/to/model")

    @patch("builtins.open", mock_open(read_data="Save quantized model to /path/to/task_workspace/task_id/model/1.pb."))
    def test_get_q_model_path_success_task_id(self):
        log_path = "fake_log_path"
        q_model_path = get_q_model_path(log_path, "task_id")
        self.assertEqual(q_model_path, "/path/to/task_workspace/task_id/model")

    @patch("builtins.open", mock_open(read_data="No quantized model saved."))
    def test_get_q_model_path_failure(self):
        log_path = "fake_log_path"
        q_model_path = get_q_model_path(log_path, "task_id")
        self.assertEqual(q_model_path, "quantized model path not found")


if __name__ == "__main__":
    unittest.main()

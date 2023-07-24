import os
import shutil
import unittest
from unittest.mock import mock_open, patch

from neural_solution.frontend.utility import (
    check_log_exists,
    deserialize,
    get_baseline_during_tuning,
    get_cluster_info,
    get_cluster_table,
    get_res_during_tuning,
    list_to_string,
    serialize,
)

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
DB_PATH = NEURAL_SOLUTION_WORKSPACE + "/db/task.db"
TASK_WORKSPACE = NEURAL_SOLUTION_WORKSPACE + "/task_workspace"
TASK_LOG_path = NEURAL_SOLUTION_WORKSPACE + "/task_log"
SERVE_LOG_PATH = NEURAL_SOLUTION_WORKSPACE + "/serve_log"


class TestMyModule(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not os.path.exists(TASK_LOG_path):
            os.makedirs(TASK_LOG_path)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(NEURAL_SOLUTION_WORKSPACE, ignore_errors=True)

    def test_serialize(self):
        request = {"key": "value"}
        expected_result = b'{"key": "value"}'
        self.assertEqual(serialize(request), expected_result)

    def test_deserialize(self):
        request = b'{"key": "value"}'
        expected_result = {"key": "value"}
        self.assertEqual(deserialize(request), expected_result)

    @patch("sqlite3.connect")
    def test_get_cluster_info(self, mock_connect):
        mock_cursor = mock_connect().cursor.return_value
        mock_cursor.fetchall.return_value = [(1, "node info", "status", 1, 2, 3)]
        expected_result = {"Cluster info": [(1, "node info", "status", 1, 2, 3)]}
        self.assertEqual(get_cluster_info(TASK_LOG_path), expected_result)

    @patch("sqlite3.connect")
    def test_get_cluster_table(self, mock_connect):
        mock_cursor = mock_connect().cursor.return_value
        mock_cursor.fetchall.return_value = [(1, "node info", "status", 1, 2, 3)]
        expected_result = (
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th>Node</th>\n"
            "      <th>Node info</th>\n"
            "      <th>status</th>\n"
            "      <th>free workers</th>\n"
            "      <th>busy workers</th>\n"
            "      <th>total workers</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <td>1</td>\n"
            "      <td>node info</td>\n"
            "      <td>status</td>\n"
            "      <td>1</td>\n"
            "      <td>2</td>\n"
            "      <td>3</td>\n"
            "    </tr>\n"
            "  </tbody>\n"
            "</table>"
        )
        self.assertEqual(get_cluster_table(TASK_LOG_path), expected_result)

    def test_get_res_during_tuning(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with open(log_path, "w") as f:
            f.write("Tune 1 result is: (int8|fp32): 0.123 (int8|fp32): 0.456")
        expected_result = {"Tuning count": "1", "Accuracy": "0.123", "Duration (seconds)": "0.456"}
        self.assertEqual(get_res_during_tuning(task_id, TASK_LOG_path), expected_result)
        os.remove(log_path)

    def test_get_baseline_during_tuning(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with open(log_path, "w") as f:
            f.write("FP32 baseline is: 0.123 0.456")
        expected_result = {"Accuracy": "0.123", "Duration (seconds)": "0.456"}
        self.assertEqual(get_baseline_during_tuning(task_id, TASK_LOG_path), expected_result)
        os.remove(log_path)

    def test_check_log_exists(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            self.assertTrue(check_log_exists(task_id, TASK_LOG_path))
            mock_exists.return_value = False
            self.assertFalse(check_log_exists(task_id, TASK_LOG_path))

    def test_list_to_string(self):
        lst = ["Hello", "Neural", "Solution"]
        expected_result = "Hello Neural Solution"
        self.assertEqual(list_to_string(lst), expected_result)


if __name__ == "__main__":
    unittest.main()

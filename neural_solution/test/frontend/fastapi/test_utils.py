import unittest
import os
import shutil
from unittest.mock import patch, mock_open

from frontend.fastapi.utils import serialize, deserialize, get_config, get_cluster_info,\
    get_cluster_table, get_res_during_tuning, get_baseline_during_tuning, check_log_exists
from frontend.fastapi.utils import TASK_LOG_path, INC_SERVE_WORKSPACE


class TestMyModule(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        if not os.path.exists(TASK_LOG_path):
            os.makedirs(TASK_LOG_path)
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree(INC_SERVE_WORKSPACE, ignore_errors=True)
        
    def test_serialize(self):
        request = {"key": "value"}
        expected_result = b'{"key": "value"}'
        self.assertEqual(serialize(request), expected_result)

    def test_deserialize(self):
        request = b'{"key": "value"}'
        expected_result = {"key": "value"}
        self.assertEqual(deserialize(request), expected_result)

    @patch("builtins.open", new_callable=mock_open, read_data='TASK_MONITOR_PORT = 2222\nRESULT_MONITOR_PORT = 3333\n')
    def test_get_config(self, mock_file):
        expected_result = (2222, 3333)
        self.assertEqual(get_config(), expected_result)

    @patch('sqlite3.connect')
    def test_get_cluster_info(self, mock_connect):
        mock_cursor = mock_connect().cursor.return_value
        mock_cursor.fetchall.return_value = [(1, "node info", "status", 1, 2, 3)]
        expected_result = {"Cluster info": [(1, "node info", "status", 1, 2, 3)]}
        self.assertEqual(get_cluster_info(), expected_result)

    @patch('sqlite3.connect')
    def test_get_cluster_table(self, mock_connect):
        mock_cursor = mock_connect().cursor.return_value
        mock_cursor.fetchall.return_value = [(1, "node info", "status", 1, 2, 3)]
        expected_result = ('<table border="1" class="dataframe">\n'
                    '  <thead>\n'
                    '    <tr style="text-align: right;">\n'
                    '      <th>Node</th>\n'
                    '      <th>Node info</th>\n'
                    '      <th>status</th>\n'
                    '      <th>free workers</th>\n'
                    '      <th>busy workers</th>\n'
                    '      <th>total workers</th>\n'
                    '    </tr>\n'
                    '  </thead>\n'
                    '  <tbody>\n'
                    '    <tr>\n'
                    '      <td>1</td>\n'
                    '      <td>node info</td>\n'
                    '      <td>status</td>\n'
                    '      <td>1</td>\n'
                    '      <td>2</td>\n'
                    '      <td>3</td>\n'
                    '    </tr>\n'
                    '  </tbody>\n'
                    '</table>')
        self.assertEqual(get_cluster_table(), expected_result)

    def test_get_res_during_tuning(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with open(log_path, "w") as f:
            f.write("Tune 1 result is: (int8|fp32): 0.123 (int8|fp32): 0.456")
        expected_result = {"Tuning count": "1", "Accuracy": "0.123", "Duration (seconds)": "0.456"}
        self.assertEqual(get_res_during_tuning(task_id), expected_result)
        os.remove(log_path)

    def test_get_baseline_during_tuning(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with open(log_path, "w") as f:
            f.write("FP32 baseline is: 0.123 0.456")
        expected_result = {"Accuracy": "0.123", "Duration (seconds)": "0.456"}
        self.assertEqual(get_baseline_during_tuning(task_id), expected_result)
        os.remove(log_path)

    def test_check_log_exists(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            self.assertTrue(check_log_exists(task_id))
            mock_exists.return_value = False
            self.assertFalse(check_log_exists(task_id))

if __name__ == '__main__':
    unittest.main()
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from neural_solution.backend.result_monitor import ResultMonitor


class TestResultMonitor(unittest.TestCase):
    @patch("socket.socket")
    def test_wait_result(self, mock_socket):
        # Mock data for testing
        task_db = MagicMock()
        task_db.lookup_task_status.return_value = "COMPLETED"
        result = {"task_id": 1, "q_model_path": "path/to/q_model", "result": 0.8}
        serialized_result = json.dumps(result)

        mock_c = MagicMock()
        mock_c.recv.return_value = serialized_result

        mock_socket.return_value.accept.return_value = (mock_c, MagicMock())
        mock_socket.return_value.recv.return_value = serialized_result
        mock_socket.return_value.__enter__.return_value = mock_socket.return_value

        # Create a ResultMonitor object and call the wait_result method
        result_monitor = ResultMonitor(8080, task_db)
        with patch("neural_solution.backend.result_monitor.deserialize", return_value={"ping": "test"}):
            adding_abort = threading.Thread(
                target=result_monitor.wait_result,
                args=(),
                daemon=True,
            )
            adding_abort.start()
            adding_abort.join(timeout=0.1)

    def test_query_task_status(self):
        # Mock data for testing
        task_db = MagicMock()
        task_db.lookup_task_status.return_value = "done"

        # Create a ResultMonitor object and call the query_task_status method
        result_monitor = ResultMonitor(8080, task_db)
        result_monitor.query_task_status(1)

        # Assert that the task_db.lookup_task_status method was called with the correct argument
        task_db.lookup_task_status.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

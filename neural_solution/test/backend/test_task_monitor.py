import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from neural_solution.backend.task import Task
from neural_solution.backend.task_monitor import TaskMonitor


class TestTaskMonitor(unittest.TestCase):
    def setUp(self):
        self.mock_task_db = Mock()
        self.mock_socket = Mock()
        self.task_monitor = TaskMonitor(8888, self.mock_task_db)
        self.task_monitor.s = self.mock_socket

    def test__start_listening(self):
        mock_bind = MagicMock()
        mock_listen = MagicMock()
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.bind = mock_bind
            mock_socket.return_value.listen = mock_listen
            self.task_monitor._start_listening("localhost", 8888, 10)

    def test_receive_task(self):
        self.mock_socket.accept.return_value = (
            Mock(),
            b'{"task_id": 123, "arguments": {}, "workers": 1, \
            "status": "pending", "script_url": "http://example.com", "optimized": True, \
            "approach": "static", "requirement": "neural_solution", "result": "", "q_model_path": ""}',
        )
        self.mock_task_db.get_task_by_id.return_value = Task(
            task_id=123,
            arguments={},
            workers=1,
            status="pending",
            script_url="http://example.com",
            optimized=True,
            approach="static",
            requirement="neural_solution",
            result="",
            q_model_path="",
        )

        # Test normal task case
        with patch(
            "neural_solution.backend.task_monitor.deserialize",
            return_value={
                "task_id": 123,
                "arguments": {},
                "workers": 1,
                "status": "pending",
                "script_url": "http://example.com",
                "optimized": True,
                "approach": "static",
                "requirement": "neural_solution",
                "result": "",
                "q_model_path": "",
            },
        ):
            task = self.task_monitor._receive_task()
            self.assertEqual(task.task_id, 123)
            self.mock_task_db.get_task_by_id.assert_called_once_with(123)

        # Test ping case
        with patch("neural_solution.backend.task_monitor.deserialize", return_value={"ping": "test"}):
            response = self.task_monitor._receive_task()
            self.assertEqual(response, False)
            self.mock_task_db.get_task_by_id.assert_called_once_with(123)

    def test_append_task(self):
        task = Task(
            task_id=123,
            arguments={},
            workers=1,
            status="pending",
            script_url="http://example.com",
            optimized=True,
            approach="static",
            requirement="neural_solution",
            result="",
            q_model_path="",
        )
        self.task_monitor._append_task(task)
        self.mock_task_db.append_task.assert_called_once_with(task)

    def test_wait_new_task(self):
        # Set up mock objects
        mock_logger = MagicMock()
        mock_task = MagicMock()
        mock_receive_task = MagicMock(return_value=mock_task)
        mock_append_task = MagicMock()
        self.task_monitor._receive_task = mock_receive_task
        self.task_monitor._append_task = mock_append_task
        self.task_monitor._start_listening = MagicMock()

        # Call the function to be tested
        adding_abort = threading.Thread(
            target=self.task_monitor.wait_new_task,
            args=(),
            daemon=True,
        )
        adding_abort.start()
        adding_abort.join(timeout=1)

        # Test task is False
        mock_receive_task = MagicMock(return_value=False)
        mock_append_task = MagicMock()
        self.task_monitor._receive_task = mock_receive_task

        adding_abort.join(timeout=1)


if __name__ == "__main__":
    unittest.main()

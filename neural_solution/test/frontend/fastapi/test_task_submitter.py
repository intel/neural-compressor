import socket
import unittest
from unittest.mock import patch

from neural_solution.frontend.task_submitter import Task, TaskSubmitter


class TestTask(unittest.TestCase):
    def test_task_creation(self):
        script_url = "https://example.com"
        optimized = True
        arguments = ["arg1", "arg2"]
        approach = "approach"
        requirements = ["req1", "req2"]
        workers = 2

        task = Task(
            script_url=script_url,
            optimized=optimized,
            arguments=arguments,
            approach=approach,
            requirements=requirements,
            workers=workers,
        )

        self.assertEqual(task.script_url, script_url)
        self.assertEqual(task.optimized, optimized)
        self.assertEqual(task.arguments, arguments)
        self.assertEqual(task.approach, approach)
        self.assertEqual(task.requirements, requirements)
        self.assertEqual(task.workers, workers)


class TestTaskSubmitter(unittest.TestCase):
    @patch("socket.socket")
    def test_submit_task(self, mock_socket):
        task_submitter = TaskSubmitter()
        task_id = "1234"
        task_submitter.submit_task(task_id)
        mock_socket.return_value.connect.assert_called_once_with(("localhost", 2222))
        mock_socket.return_value.send.assert_called_once_with(b'{"task_id": "1234"}')
        mock_socket.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

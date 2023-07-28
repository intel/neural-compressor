import unittest

from neural_solution.backend.task import Task


class TestTask(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            "123", "python script.py", 4, "pending", "http://example.com/script.py", True, "approach", "requirement"
        )

    def test_task_attributes(self):
        self.assertEqual(self.task.task_id, "123")
        self.assertEqual(self.task.arguments, "python script.py")
        self.assertEqual(self.task.workers, 4)
        self.assertEqual(self.task.status, "pending")
        self.assertEqual(self.task.script_url, "http://example.com/script.py")
        self.assertEqual(self.task.optimized, True)
        self.assertEqual(self.task.approach, "approach")
        self.assertEqual(self.task.requirement, "requirement")
        self.assertEqual(self.task.result, "")
        self.assertEqual(self.task.q_model_path, "")


if __name__ == "__main__":
    unittest.main()

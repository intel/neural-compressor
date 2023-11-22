import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from neural_solution.backend.task_db import Task, TaskDB
from neural_solution.utils.utility import get_db_path

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
db_path = get_db_path(NEURAL_SOLUTION_WORKSPACE)


class TestTaskDB(unittest.TestCase):
    def setUp(self):
        self.taskdb = TaskDB(db_path=db_path)
        self.task = Task(
            "1", "arguments", 1, "pending", "script_url", 0, "approach", "requirement", "result", "q_model_path"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("ns_workspace")

    def test_append_task(self):
        self.taskdb.append_task(self.task)
        self.assertEqual(len(self.taskdb.task_queue), 1)
        self.assertEqual(self.taskdb.task_queue[0], "1")

    def test_get_pending_task_num(self):
        self.taskdb.append_task(self.task)
        self.assertEqual(self.taskdb.get_pending_task_num(), 1)

    def test_get_all_pending_tasks(self):
        self.taskdb.cursor.execute(
            "insert into task values ('2', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        pending_tasks = self.taskdb.get_all_pending_tasks()
        self.assertEqual(len(pending_tasks), 1)
        self.assertEqual(pending_tasks[0].task_id, "2")

    def test_update_task_status(self):
        self.taskdb.cursor.execute(
            "insert into task values ('3', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        self.taskdb.update_task_status("3", "running")
        self.taskdb.cursor.execute("select status from task where id='3'")
        status = self.taskdb.cursor.fetchone()[0]
        self.assertEqual(status, "running")
        with self.assertRaises(Exception):
            self.taskdb.update_task_status("3", "invalid_status")

    def test_update_result(self):
        self.taskdb.cursor.execute(
            "insert into task values ('4', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        self.taskdb.update_result("4", "new_result")
        self.taskdb.cursor.execute("select result from task where id='4'")
        result = self.taskdb.cursor.fetchone()[0]
        self.assertEqual(result, "new_result")

    def test_update_q_model_path_and_result(self):
        self.taskdb.cursor.execute(
            "insert into task values ('5', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        self.taskdb.update_q_model_path_and_result("5", "new_q_model_path", "new_result")
        self.taskdb.cursor.execute("select q_model_path, result from task where id='5'")
        q_model_path, result = self.taskdb.cursor.fetchone()
        self.assertEqual(q_model_path, "new_q_model_path")
        self.assertEqual(result, "new_result")

    def test_lookup_task_status(self):
        self.taskdb.cursor.execute(
            "insert into task values ('6', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        status_dict = self.taskdb.lookup_task_status("6")
        self.assertEqual(status_dict["status"], "pending")
        self.assertEqual(status_dict["result"], "result")

    def test_get_task_by_id(self):
        self.taskdb.cursor.execute(
            "insert into task values ('7', 'arguments', 1, \
            'pending', 'script_url', 0, 'approach', 'requirement', 'result', 'q_model_path')"
        )
        self.taskdb.conn.commit()
        task = self.taskdb.get_task_by_id("7")
        self.assertEqual(task.task_id, "7")
        self.assertEqual(task.arguments, "arguments")
        self.assertEqual(task.workers, 1)
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.result, "result")

    def test_remove_task(self):
        self.taskdb.remove_task("1")
        # currently no garbage collection, so this function does nothing


if __name__ == "__main__":
    unittest.main()

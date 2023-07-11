import os
import shutil
import threading
import unittest
from subprocess import CalledProcessError
from unittest.mock import MagicMock, Mock, mock_open, patch

from neural_solution.backend.cluster import Cluster
from neural_solution.backend.scheduler import Scheduler
from neural_solution.backend.task import Task
from neural_solution.backend.task_db import TaskDB
from neural_solution.backend.utils.utility import dump_elapsed_time, get_task_log_path
from neural_solution.config import config
from neural_solution.utils.utility import get_db_path

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
db_path = get_db_path(NEURAL_SOLUTION_WORKSPACE)
config.workspace = NEURAL_SOLUTION_WORKSPACE


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.cluster = Cluster(db_path=db_path)
        self.task_db = TaskDB(db_path=db_path)
        self.result_monitor_port = 1234
        self.scheduler = Scheduler(
            self.cluster, self.task_db, self.result_monitor_port, conda_env_name="for_ns_test", config=config
        )

    def tearDown(self) -> None:
        shutil.rmtree("ns_workspace", ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("examples")

    def test_prepare_env(self):
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "pip",
            "test_result",
            "test_q_model_path",
        )
        result = self.scheduler.prepare_env(task)
        self.assertTrue(result.startswith(self.scheduler.conda_env_name))

        # Test requirement in {conda_env} case
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "pip",
            "test_result",
            "test_q_model_path",
        )
        scheduler_test = Scheduler(
            self.cluster, self.task_db, self.result_monitor_port, conda_env_name="base", config=config
        )
        result = scheduler_test.prepare_env(task)
        self.assertTrue(result.startswith("base"))

        # Test requirement is '' case
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "",
            "test_result",
            "test_q_model_path",
        )
        result = self.scheduler.prepare_env(task)
        self.assertEqual(result, self.scheduler.conda_env_name)

    def test_prepare_task(self):
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_example",
            "test_optimized",
            "static",
            "test_requirement",
            "test_result",
            "test_q_model_path",
        )
        test_path = "examples/test_example"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        with open(os.path.join(test_path, "test.py"), "w") as f:
            f.write("# For Test")

        self.scheduler.prepare_task(task)

        # url case
        with patch("neural_solution.backend.scheduler.is_remote_url", return_value=True):
            self.scheduler.prepare_task(task)

        # optimized is False case
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_example",
            False,
            "static",
            "test_requirement",
            "test_result",
            "test_q_model_path",
        )
        self.scheduler.prepare_task(task)

        with patch("neural_solution.backend.scheduler.is_remote_url", return_value=True):
            task = Task(
                "test_task",
                "test_arguments",
                "test_workers",
                "test_status",
                "test_example/test.py",
                False,
                "static",
                "test_requirement",
                "test_result",
                "test_q_model_path",
            )
            self.scheduler.prepare_task(task)

    def test_check_task_status(self):
        log_path = "test_log_path"
        # done case
        with patch("builtins.open", mock_open(read_data="[INFO] Save deploy yaml to\n")) as mock_file:
            result = self.scheduler.check_task_status(log_path)
            self.assertEqual(result, "done")

        # failed case
        with patch("builtins.open", mock_open(read_data="[INFO] Deploying...\n")) as mock_file:
            result = self.scheduler.check_task_status(log_path)
            self.assertEqual(result, "failed")

    def test_sanitize_arguments(self):
        arguments = "test_arguments\xa0"
        sanitized_arguments = self.scheduler.sanitize_arguments(arguments)
        self.assertEqual(sanitized_arguments, "test_arguments ")

    def test_dispatch_task(self):
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "test_requirement",
            "test_result",
            "test_q_model_path",
        )
        resource = [("node1", "8"), ("node2", "8")]
        with patch("neural_solution.backend.scheduler.Scheduler.launch_task") as mock_launch_task:
            self.scheduler.dispatch_task(task, resource)
        self.assertTrue(mock_launch_task.called)

    @patch("socket.socket")
    @patch("builtins.open")
    def test_report_result(self, mock_open, mock_socket):
        task_id = "test_task"
        log_path = "test_log_path"
        task_runtime = 10
        self.scheduler.q_model_path = None
        mock_socket.return_value.connect.return_value = None
        mock_open.return_value.readlines.return_value = ["Tune 1 result is: (int8|fp32): 0.8 (int8|fp32): 0.9"]
        expected_result = {"optimization time (seconds)": "10.00", "Accuracy": "0.8", "Duration (seconds)": "0.9"}

        self.scheduler.report_result(task_id, log_path, task_runtime)

        mock_open.assert_called_once_with(log_path)
        mock_socket.assert_called_once_with()
        mock_socket.return_value.connect.assert_called_once_with(("localhost", 1234))
        mock_socket.return_value.send.assert_called_once()

    @patch("neural_solution.backend.scheduler.Scheduler.prepare_task")
    @patch("neural_solution.backend.scheduler.Scheduler.prepare_env")
    @patch("neural_solution.backend.scheduler.Scheduler._parse_cmd")
    @patch("subprocess.Popen")
    @patch("neural_solution.backend.scheduler.Scheduler.check_task_status")
    @patch("neural_solution.backend.scheduler.Scheduler.report_result")
    def test_launch_task(
        self,
        mock_report_result,
        mock_check_task_status,
        mock_popen,
        mock_parse_cmd,
        mock_prepare_env,
        mock_prepare_task,
    ):
        task = Task(
            "test_task",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "test_requirement",
            "test_result",
            "test_q_model_path",
        )
        resource = ["1 node1", "2 node2"]
        mock_parse_cmd.return_value = "test_cmd"
        mock_check_task_status.return_value = "done"
        mock_popen.return_value.wait.return_value = None
        mock_prepare_env.return_value = "test_env"
        mock_prepare_task.return_value = None
        mock_report_result.return_value = None

        self.scheduler.launch_task(task, resource)

    @patch("neural_solution.backend.scheduler.Scheduler.launch_task")
    @patch("neural_solution.backend.cluster.Cluster.reserve_resource")
    def test_schedule_tasks(self, mock_reserve_resource, mock_launch_task):
        task1 = Task(
            "1",
            "test_arguments",
            "test_workers",
            "test_status",
            "test_script_url",
            "test_optimized",
            "test_approach",
            "test_requirement",
            "test_result",
            "test_q_model_path",
        )
        self.task_db.cursor.execute(
            "insert into task values ('1', 'test_arguments', 'test_workers', \
            'test_status', 'test_script_url', \
            'test_optimized', 'test_approach', 'test_requirement', 'test_result', 'test_q_model_path')"
        )

        # no pending task case
        adding_abort = threading.Thread(
            target=self.scheduler.schedule_tasks,
            args=(),
            daemon=True,
        )
        adding_abort.start()
        adding_abort.join(timeout=10)

        # task case
        self.task_db.append_task(task1)
        mock_reserve_resource.return_value = [("node1", 8)]
        mock_launch_task.return_value = None

        adding_abort = threading.Thread(
            target=self.scheduler.schedule_tasks,
            args=(),
            daemon=True,
        )
        adding_abort.start()
        adding_abort.join(timeout=10)

        # no resource case
        self.task_db.append_task(task1)
        mock_reserve_resource.return_value = False
        adding_abort = threading.Thread(
            target=self.scheduler.schedule_tasks,
            args=(),
            daemon=True,
        )
        adding_abort.start()
        adding_abort.join(timeout=10)


class TestParseCmd(unittest.TestCase):
    def setUp(self):
        self.cluster = Cluster(db_path=db_path)
        self.task_db = TaskDB(db_path=db_path)
        self.result_monitor_port = 1234
        self.task_scheduler = Scheduler(
            self.cluster, self.task_db, self.result_monitor_port, conda_env_name="for_ns_test", config=config
        )
        self.task = MagicMock()
        self.resource = ["1 node1", "2 node2", "3 node3"]
        self.task.workers = 3
        self.task_name = "test_task"
        self.script_name = "test_script.py"
        self.task_path = "/path/to/task"
        self.arguments = "arg1 arg2"
        self.task.arguments = self.arguments
        self.task.name = self.task_name
        self.task.optimized = True
        self.task.script = self.script_name
        self.task.task_path = self.task_path
        self.task_scheduler.script_name = self.script_name
        self.task_scheduler.task_path = self.task_path

    def test__parse_cmd(self):
        expected_cmd = (
            "cd /path/to/task\nmpirun -np 3 -host node1,node2,node3 -map-by socket:pe=5"
            + " -mca btl_tcp_if_include 192.168.20.0/24 -x OMP_NUM_THREADS=5 --report-bindings bash distributed_run.sh"
        )
        with patch("neural_solution.backend.scheduler.Scheduler.prepare_task") as mock_prepare_task, patch(
            "neural_solution.backend.scheduler.Scheduler.prepare_env"
        ) as mock_prepare_env, patch("neural_solution.backend.scheduler.logger.info") as mock_logger_info, patch(
            "builtins.open", create=True
        ) as mock_open, patch(
            "neural_solution.backend.scheduler.os.path.join"
        ) as mock_os_path_join:
            mock_prepare_task.return_value = None
            mock_prepare_env.return_value = "test_env"
            mock_logger_info.return_value = None
            mock_open.return_value.__enter__ = lambda x: x
            mock_open.return_value.__exit__ = MagicMock()
            mock_os_path_join.return_value = "/path/to/task/distributed_run.sh"

            result = self.task_scheduler._parse_cmd(self.task, self.resource)
            self.assertEqual(result, expected_cmd)

            mock_prepare_task.assert_called_once_with(self.task)
            mock_prepare_env.assert_called_once_with(self.task)
            mock_logger_info.assert_called_once_with("[TaskScheduler] host resource: node1,node2,node3")
            mock_open.assert_called_once_with("/path/to/task/distributed_run.sh", "w", encoding="utf-8")
            mock_os_path_join.assert_called_once_with("/path/to/task", "distributed_run.sh")

            self.task.optimized = False
            result = self.task_scheduler._parse_cmd(self.task, self.resource)
            self.assertEqual(result, expected_cmd)


if __name__ == "__main__":
    unittest.main()

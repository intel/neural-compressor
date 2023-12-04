import asyncio
import os
import shutil
import sqlite3
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastapi import WebSocket
from fastapi.testclient import TestClient

from neural_solution.config import config
from neural_solution.frontend.fastapi.main_server import LogEventHandler, Observer, app, start_log_watcher

NEURAL_SOLUTION_WORKSPACE = os.path.join(os.getcwd(), "ns_workspace")
DB_PATH = NEURAL_SOLUTION_WORKSPACE + "/db"
TASK_WORKSPACE = NEURAL_SOLUTION_WORKSPACE + "/task_workspace"
TASK_LOG_path = NEURAL_SOLUTION_WORKSPACE + "/task_log"
SERVE_LOG_PATH = NEURAL_SOLUTION_WORKSPACE + "/serve_log"

client = TestClient(app)


def build_db():
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    conn = sqlite3.connect(
        f"{DB_PATH}/task.db", check_same_thread=False
    )  # sqlite should set this check_same_thread to False
    cursor = conn.cursor()
    cursor.execute(
        "create table if not exists task(id TEXT PRIMARY KEY, arguments varchar(100), "
        + "workers int, status varchar(20), script_url varchar(500), optimized integer, "
        + "approach varchar(20), requirements varchar(500), result varchar(500), q_model_path varchar(200))"
    )
    cursor.execute("drop table if exists cluster ")
    cursor.execute(
        r"create table cluster(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        + "node_info varchar(500),"
        + "status varchar(100),"
        + "free_sockets int,"
        + "busy_sockets int,"
        + "total_sockets int)"
    )

    conn.commit()
    conn.close


def delete_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


def use_db():
    def f(func):
        def fi(*args, **kwargs):
            build_db()
            res = func(*args, **kwargs)
            delete_db()

        return fi

    return f


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not os.path.exists(TASK_LOG_path):
            os.makedirs(TASK_LOG_path)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(NEURAL_SOLUTION_WORKSPACE, ignore_errors=True)
        delete_db()

    def test_read_root(self):
        response = client.get("/")
        assert response.status_code == 200
        self.assertEqual(response.json(), {"message": "Welcome to Neural Solution!"})

    @patch("neural_solution.frontend.fastapi.main_server.socket")
    def test_ping(self, mock_socket):
        response = client.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("msg", response.json())

    @use_db()
    def test_get_cluster(self):
        response = client.get("/cluster")
        assert response.status_code == 200
        assert "Cluster info" in response.json()

    @use_db()
    def test_get_clusters(self):
        response = client.get("/clusters")
        assert response.status_code == 200
        assert "table" in response.text

    def test_get_description(self):
        data = {
            "description": "",
        }
        path = "../../doc"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "user_facing_api.json"), "w") as f:
            import json

            json.dump(data, f)
        response = client.get("/description")
        assert response.status_code == 200
        assert "description" in response.text
        shutil.rmtree(path)

    @patch("neural_solution.frontend.fastapi.main_server.task_submitter.submit_task")
    def test_submit_task(self, mock_submit_task):
        task = {
            "script_url": "http://example.com/script.py",
            "optimized": True,
            "arguments": ["arg1", "arg2"],
            "approach": "approach1",
            "requirements": ["req1", "req2"],
            "workers": 3,
        }

        # test no db case
        delete_db()
        response = client.post("/task/submit/", json=task)
        self.assertEqual(response.status_code, 200)
        self.assertIn("msg", response.json())
        self.assertIn("Task Submitted fail! db not found!", response.json()["msg"])
        mock_submit_task.assert_not_called()

        # test successfully
        build_db()
        response = client.post("/task/submit/", json=task)
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("task_id", response.json())
        self.assertIn("msg", response.json())
        self.assertIn("successfully", response.json()["status"])
        mock_submit_task.assert_called_once()

        # test ConnectionRefusedError case
        mock_submit_task.side_effect = ConnectionRefusedError
        response = client.post("/task/submit/", json=task)
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("task_id", response.json())
        self.assertIn("msg", response.json())
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("Task Submitted fail! Make sure Neural Solution runner is running!", response.json()["msg"])
        mock_submit_task.assert_called()

        # test generic Exception case
        mock_submit_task.side_effect = Exception("Something went wrong")
        response = client.post("/task/submit/", json=task)
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("task_id", response.json())
        self.assertIn("msg", response.json())
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("Something went wrong", response.json()["msg"])
        mock_submit_task.assert_called()

        delete_db()

    @use_db()
    @patch("neural_solution.frontend.fastapi.main_server.task_submitter.submit_task")
    def test_get_task_by_id(self, mock_submit_task):
        task = {
            "script_url": "http://example.com/script.py",
            "optimized": True,
            "arguments": ["arg1", "arg2"],
            "approach": "approach1",
            "requirements": ["req1", "req2"],
            "workers": 3,
        }
        response = client.post("/task/submit/", json=task)
        task_id = response.json()["task_id"]
        response = client.get(f"/task/{task_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    @use_db()
    def test_get_all_tasks(self):
        response = client.get("/task/")
        assert response.status_code == 200
        delete_db()
        response = client.get("/task/")
        assert response.status_code == 200
        assert response.json()["message"] is None

    @use_db()
    @patch("neural_solution.frontend.fastapi.main_server.task_submitter.submit_task")
    def test_get_task_status_by_id(self, mock_submit_task):
        task = {
            "script_url": "http://example.com/script.py",
            "optimized": True,
            "arguments": ["arg1", "arg2"],
            "approach": "approach1",
            "requirements": ["req1", "req2"],
            "workers": 3,
        }
        response = client.post("/task/submit/", json=task)
        task_id = response.json()["task_id"]
        response = client.get(f"/task/status/{task_id}")
        assert response.status_code == 200
        self.assertIn("pending", response.text)

        response = client.get("/task/status/error_id")
        assert response.status_code == 200
        self.assertIn("Please check url", response.text)

    def test_read_logs(self):
        task_id = "12345"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        with open(log_path, "w") as f:
            f.write(f"I am {task_id}.")
        response = client.get(f"/task/log/{task_id}")
        assert response.status_code == 200
        self.assertIn(task_id, response.text)
        os.remove(log_path)


class TestLogEventHandler(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_init(self):
        mock_websocket = MagicMock()
        mock_websocket.send_text = MagicMock()
        handler = LogEventHandler(mock_websocket, "test_task_id", 0)
        self.assertEqual(handler.websocket, mock_websocket)
        self.assertEqual(handler.task_id, "test_task_id")
        self.assertEqual(handler.last_position, 0)
        self.assertIsInstance(handler.queue, asyncio.Queue)
        self.assertIsInstance(handler.timer, asyncio.Task)

    def test_on_modified(self):
        config.workspace = NEURAL_SOLUTION_WORKSPACE
        mock_websocket = MagicMock()
        mock_websocket.send_text = MagicMock()
        task_id = "1234"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        if not os.path.exists(TASK_LOG_path):
            os.makedirs(TASK_LOG_path)

        with open(log_path, "w") as f:
            f.write(f"I am {task_id}.")

        handler = LogEventHandler(mock_websocket, "1234", 0)

        handler.queue.put_nowait("test message")
        event = MagicMock()
        task_id = "1234"
        log_path = f"{TASK_LOG_path}/task_{task_id}.txt"
        event.src_path = log_path
        with patch("builtins.open", MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.seek.return_value = None
            mock_file.return_value.__enter__.return_value.readlines.return_value = ["test line"]
            handler.on_modified(event)
            mock_file.assert_called_once_with(log_path, "r")
            mock_file.return_value.__enter__.return_value.seek.assert_called_once_with(0)
            mock_file.return_value.__enter__.return_value.readlines.assert_called_once()
            # handler.queue.put_nowait.assert_called_once_with("test line")
        os.remove(log_path)


class TestStartLogWatcher(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_start_log_watcher(self):
        mock_observer = MagicMock()
        mock_observer.schedule = MagicMock()
        with patch("neural_solution.frontend.fastapi.main_server.Observer", MagicMock(return_value=mock_observer)):
            observer = start_log_watcher("test_websocket", "1234", 0)
            self.assertIsInstance(observer, type(mock_observer))


class TestWebsocketEndpoint(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.client = TestClient(app)

    def test_websocket_endpoint(self):
        pass
        # with self.assertRaises(HTTPException):
        #     asyncio.run(websocket_endpoint(WebSocket, "nonexistent_task"))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural Solution task database."""
import sqlite3
import threading
from collections import deque

from neural_solution.backend.task import Task
from neural_solution.backend.utils.utility import create_dir


class TaskDB:
    """TaskDb manages all the tasks.

    TaskDb provides atomic operations on managing the task queue and task details.

    Attributes:
        task_queue: a FIFO queue that only holds pending task ids
        task_collections: a growing-only list of all task objects and their details (no garbage collection currently)
        lock: the lock on the data structures to provide atomic operations
    """

    def __init__(self, db_path):
        """Init TaskDB.

        Args:
            db_path (str): the database path.
        """
        self.task_queue = deque()
        create_dir(db_path)
        # sqlite should set this check_same_thread to False
        self.conn = sqlite3.connect(f"{db_path}", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "create table if not exists task(id TEXT PRIMARY KEY, arguments varchar(100), "
            + "workers int, status varchar(20), script_url varchar(500), optimized integer, "
            + "approach varchar(20), requirements varchar(500), result varchar(500), q_model_path varchar(200))"
        )
        self.conn.commit()
        # self.task_collections = []
        self.lock = threading.Lock()

    def append_task(self, task):
        """Append the task to the task queue."""
        with self.lock:
            self.task_queue.append(task.task_id)

    def get_pending_task_num(self):
        """Get the number of the pending tasks."""
        with self.lock:
            return len(self.task_queue)

    def get_all_pending_tasks(self):
        """Get all the pending task objects."""
        self.cursor.execute(r"select * from task where status=='pending'")
        task_lst = self.cursor.fetchall()
        res_lst = []
        for task_tuple in task_lst:
            res_lst.append(Task(*task_tuple))
        return res_lst

    def update_task_status(self, task_id, status):
        """Update the task status with the task id and the status."""
        if status not in ["pending", "running", "done", "failed"]:
            raise Exception("status invalid, should be one of pending/running/done")
        self.cursor.execute(r"update task set status='{}' where id=?".format(status), (task_id,))
        self.conn.commit()

    def update_result(self, task_id, result_str):
        """Update the task result with the result string."""
        self.cursor.execute(r"update task set result='{}' where id={}".format(result_str, task_id))
        self.conn.commit()

    def update_q_model_path_and_result(self, task_id, q_model_path, result_str):
        """Update the task result with the result string."""
        self.cursor.execute(
            r"update task set q_model_path='{}', result='{}' where id=?".format(q_model_path, result_str), (task_id,)
        )
        self.conn.commit()

    def lookup_task_status(self, task_id):
        """Look up the current task status and result."""
        self.cursor.execute(r"select status, result from task where id=?", (task_id,))
        status, result = self.cursor.fetchone()
        return {"status": status, "result": result}

    def get_task_by_id(self, task_id):
        """Get the task object by task id."""
        self.cursor.execute(r"select * from task where id=?", (task_id,))
        attr_tuple = self.cursor.fetchone()
        return Task(*attr_tuple)

    def remove_task(self, task_id):  # currently no garbage collection
        """Remove task."""
        pass

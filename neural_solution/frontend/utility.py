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
"""Common utilities for all frontend components."""

import json
import os
import re
import socket
import sqlite3
import uuid

import pandas as pd

from neural_solution.utils import logger
from neural_solution.utils.utility import dict_to_str, get_task_log_workspace


def query_task_status(task_id, db_path):
    """Query task status according to id.

    Args:
        task_id (str): the id of task
        db_path (str): the path of database

    Returns:
        dict: the task status and information.
    """
    res = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    return {
        "status": res[0],
        "optimized_result": dict_to_str(deserialize(res[1]) if res[1] else res[1]),
        "result_path": res[2],
    }


def query_task_result(task_id, db_path, workspace):
    """Query the task result according id.

    Args:
        task_id (str): the id of task
        db_path (str): the path of database
        workspace (str): the workspace for Neural Solution

    Returns:
        dict: task result
    """
    status = "unknown"
    tuning_info = {}
    optimization_result = {}

    res = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    logger.info("in query")
    if not res:
        status = "Please check url."
    elif res[0] == "done":
        status = res[0]
        optimization_result = deserialize(res[1]) if res[1] else res[1]
        optimization_result["result_path"] = res[2]
    elif res[0] == "pending":
        status = "pending"
    else:
        baseline = get_baseline_during_tuning(task_id, get_task_log_workspace(workspace))
        tuning_result = get_res_during_tuning(task_id, get_task_log_workspace(workspace))
        status = res[0]
        tuning_info = {"baseline": baseline, "message": tuning_result}
    result = {"status": status, "tuning_information": tuning_info, "optimization_result": optimization_result}
    return result


def check_service_status(port_lst, service_address):
    """Check server status.

    Args:
        port_lst (List): ports list
        service_address (str): service ip

    Returns:
        dict: server status and messages
    """
    count = 0
    msg = "Neural Solution is running."
    for port in port_lst:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((service_address, port))
            sock.send(serialize({"ping": "test"}))
            sock.settimeout(5)
            response = sock.recv(1024)
            if response == b"ok":
                count += 1
                sock.close()
                continue
        except ConnectionRefusedError:
            msg = "Ping fail! Make sure Neural Solution runner is running!"
            break
        except Exception as e:
            msg = "Ping fail! {}".format(e)
            break
        sock.close()
    return {"status": "Healthy", "msg": msg} if count == 1 else {"status": "Failed", "msg": msg}


def submit_task_to_db(task, task_submitter, db_path):
    """Submit the task to db.

    Args:
        task (Task): the object of Task
        task_submitter (TaskSubmitter): the object of TaskSubmitter
        db_path (str): the path of database

    Returns:
        str: task id and information
    """
    msg = "Task submitted failed"
    status = "failed"
    task_id = "-1"
    result = {"status": status, "task_id": task_id, "msg": msg}
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        task_id = str(uuid.uuid4()).replace("-", "")
        sql = (
            r"insert into task(id, script_url, optimized, arguments, approach, requirements, workers, status)"
            + r" values ('{}', '{}', {}, '{}', '{}', '{}', {}, 'pending')".format(
                task_id,
                task.script_url,
                task.optimized,
                list_to_string(task.arguments),
                task.approach,
                list_to_string(task.requirements),
                task.workers,
            )
        )
        cursor.execute(sql)
        conn.commit()
        try:
            task_submitter.submit_task(task_id)
        except ConnectionRefusedError:
            msg = "Task Submitted fail! Make sure neural solution runner is running!"
        except Exception as e:
            msg = "Task Submitted fail! {}".format(e)
        conn.close()
        status = "successfully"
        msg = "Task submitted successfully"
    else:
        msg = "Task Submitted fail! db not found!"
    result["status"] = status
    result["task_id"] = task_id
    result["msg"] = msg
    return result


def serialize(request: dict) -> bytes:
    """Serialize a dict object to bytes for inter-process communication."""
    return json.dumps(request).encode()


def deserialize(request: bytes) -> dict:
    """Deserialize the received bytes to a dict object."""
    return json.loads(request)


def get_cluster_info(db_path: str):
    """Get cluster information from database.

    Returns:
        json: cluster information includes the number of nodes and node information.
    """
    conn = sqlite3.connect(f"{db_path}")
    cursor = conn.cursor()
    cursor.execute(r"select * from cluster")
    conn.commit()
    rows = cursor.fetchall()
    conn.close()
    return {"Cluster info": rows}


def get_cluster_table(db_path: str):
    """Get cluster table from database.

    Returns:
        html: table of cluster information.
    """
    conn = sqlite3.connect(f"{db_path}")
    cursor = conn.cursor()
    cursor.execute(r"select * from cluster")
    conn.commit()
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Node", "Node info", "status", "free workers", "busy workers", "total workers"])
    html_table = df.to_html(
        index=False,
    )
    conn.close()
    return html_table


def get_res_during_tuning(task_id: str, task_log_path):
    """Get result during tuning.

    Args:
        task_id (string): used to generate log path.

    Returns:
        dict: the result of {"Tuning count":, "Accuracy":, "Duration (seconds)"}.
    """
    results = {}
    log_path = "{}/task_{}.txt".format(task_log_path, task_id)
    for line in reversed(open(log_path).readlines()):
        res_pattern = r"Tune (\d+) result is: "
        res_pattern = r"Tune (\d+) result is:\s.*?\(int8\|fp32\):\s+(\d+\.\d+).*?\(int8\|fp32\):\s+(\d+\.\d+).*?"
        res_matches = re.findall(res_pattern, line)
        if res_matches:
            results["Tuning count"] = res_matches[0][0]
            results["Accuracy"] = res_matches[0][1]
            results["Duration (seconds)"] = res_matches[0][2]
            # break when the last result is matched
            break

    logger.info("Query results: {}".format(results))
    return results if results else "Tune 1 running..."


def get_baseline_during_tuning(task_id: str, task_log_path):
    """Get result during tuning.

    Args:
        task_id (string): used to generate log path.

    Returns:
        dict: the baseline of {"Accuracy":,"Duration (seconds)":}.
    """
    results = {}
    log_path = "{}/task_{}.txt".format(task_log_path, task_id)
    for line in reversed(open(log_path).readlines()):
        res_pattern = "FP32 baseline is:\s+.*?(\d+\.\d+).*?(\d+\.\d+).*?"
        res_matches = re.findall(res_pattern, line)
        if res_matches:
            results["Accuracy"] = res_matches[0][0]
            results["Duration (seconds)"] = res_matches[0][1]
            # break when the last result is matched
            break

    logger.info("FP32 baseline: {}".format(results))
    return results if results else "Getting FP32 baseline..."


def check_log_exists(task_id: str, task_log_path):
    """Check whether the log file exists.

    Args:
        task_id (str): task id.

    Returns:
        bool: Does the log file exist.
    """
    log_path = "{}/task_{}.txt".format(task_log_path, task_id)
    if os.path.exists(log_path):
        return True
    else:
        return False


def list_to_string(lst: list):
    """Convert the list to a space concatenated string.

    Args:
        lst (list): strings

    Returns:
        str: string
    """
    return " ".join(str(i) for i in lst)

def is_valid_task(task: dict) -> bool:
    """Verify whether the task is valid.

    Args:
        task (dict): task request

    Returns:
        bool: valid or invalid
    """
    required_fields = ["script_url", "optimized", "arguments", "approach", "requirements", "workers"]

    for field in required_fields:
        if field not in task:
            return False

    if not isinstance(task["script_url"], str):
        return False

    if not isinstance(task["optimized"], str) or task["optimized"] not in ["True", "False"]:
        return False

    if not isinstance(task["arguments"], list):
        return False

    if not isinstance(task["approach"], str) or task["approach"] not in ["static", "dynamic"]:
        return False

    if not isinstance(task["requirements"], list):
        return False

    if not isinstance(task["workers"], int) or task["workers"] < 1:
        return False

    return True

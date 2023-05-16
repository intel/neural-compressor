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
import sqlite3
import os
import re
import uuid
import pandas as pd


from neural_solution.frontend.task_submitter import TaskSubmitter
from neural_solution.config import DB_PATH

# Get config from Launcher.sh
task_monitor_port = int(os.environ.get("TASK_MONITOR_PORT", 2222))
result_monitor_port = int(os.environ.get('RESULT_MONITOR_PORT', 3333))

task_submitter = TaskSubmitter(task_monitor_port=task_monitor_port, result_monitor_port=task_monitor_port)
db_path = DB_PATH

def submit_task_to_db(task):
    msg = "Task submitted failed"
    status = "failed"
    task_id = "-1"
    result = {"status": status, "task_id": task_id, "msg": msg}
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        task_id = str(uuid.uuid4()).replace('-','')
        sql = r"insert into task(id, script_url, optimized, arguments, approach, requirements, workers, status)" +\
         r" values ('{}', '{}', {}, '{}', '{}', '{}', {}, 'pending')".format(
             task_id,
             task.script_url,
             task.optimized,
             list_to_string(task.arguments),
             task.approach,
             list_to_string(task.requirements),
             task.workers)
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
    result["msg"]=msg
    return result

def serialize(request: dict) -> bytes:
    """Serialize a dict object to bytes for inter-process communication."""
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    """Deserialize the received bytes to a dict object"""
    return json.loads(request)

def get_cluster_info(db_path:str):
    """Get cluster information from database

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

def get_cluster_table(db_path:str):
    """Get cluster table from database

    Returns:
        html: table of cluster information.
    """
    conn = sqlite3.connect(f"{db_path}")
    cursor = conn.cursor()
    cursor.execute(r"select * from cluster")
    conn.commit()
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Node", "Node info", "status","free workers", "busy workers", "total workers"])
    html_table = df.to_html(index=False, )
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
        res_pattern = r'Tune (\d+) result is: '
        res_pattern = r'Tune (\d+) result is:\s.*?\(int8\|fp32\):\s+(\d+\.\d+).*?\(int8\|fp32\):\s+(\d+\.\d+).*?'
        res_matches = re.findall(res_pattern, line)
        if res_matches:
            results["Tuning count"] = res_matches[0][0]
            results["Accuracy"] = res_matches[0][1]
            results["Duration (seconds)"] = res_matches[0][2]
            # break when the last result is matched
            break

    print("Query results: {}".format(results))
    return  results if results else "Tune 1 running..."

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

    print("FP32 baseline: {}".format(results))
    return  results if results else "Getting FP32 baseline..."

def check_log_exists(task_id: str, task_log_path):
    """Check whether the log file exists

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
    return " ".join(str(i) for i in lst)
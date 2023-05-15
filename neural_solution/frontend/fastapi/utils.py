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


import json
import sqlite3
import re
import pandas as pd
import os
NEURAL_SOLUTION_WORKSPACE = os.path.abspath("../../ns_workspace")
DB_PATH = NEURAL_SOLUTION_WORKSPACE + "/db"
TASK_WORKSPACE =  NEURAL_SOLUTION_WORKSPACE + "/task_workspace"
TASK_LOG_path = NEURAL_SOLUTION_WORKSPACE + "/task_log"

def serialize(request: dict) -> bytes:
    """Serialize a dict object to bytes for inter-process communication."""
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    """Deserialize the received bytes to a dict object"""
    return json.loads(request)

def get_config():
    """Get ports from ../../backend/constant.py

    Returns:
        int, int: task monitor port & result monitor port
    """
    # task_monitor_port, result_monitor_port = 2222, 3333
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir.replace("frontend", "backend"), "constant.py")
    for line in open(path,'r').readlines():
        if "TASK_MONITOR_PORT" in line.split():
            task_monitor_port = line.split()[-1]
        if "RESULT_MONITOR_PORT" in line.split():
            result_monitor_port = line.split()[-1]
    return int(task_monitor_port), int(result_monitor_port)

def get_cluster_info():
    """Get cluster information from database

    Returns:
        json: cluster information includes the number of nodes and node information.
    """
    conn = sqlite3.connect(f"{DB_PATH}/task.db")
    cursor = conn.cursor()
    cursor.execute(r"select * from cluster")
    conn.commit()
    rows = cursor.fetchall()
    conn.close()
    return {"Cluster info": rows}

def get_cluster_table():
    """Get cluster table from database

    Returns:
        html: table of cluster information.
    """
    conn = sqlite3.connect(f"{DB_PATH}/task.db")
    cursor = conn.cursor()
    cursor.execute(r"select * from cluster")
    conn.commit()
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Node", "Node info", "status","free workers", "busy workers", "total workers"])
    html_table = df.to_html(index=False, )
    conn.close()
    return html_table

def get_res_during_tuning(task_id: str):
    """Get result during tuning.

    Args:
        task_id (string): used to generate log path.

    Returns:
        dict: the result of {"Tuning count":, "Accuracy":, "Duration (seconds)"}.
    """
    results = {}
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
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

def get_baseline_during_tuning(task_id: str):
    """Get result during tuning.

    Args:
        task_id (string): used to generate log path.

    Returns:
        dict: the baseline of {"Accuracy":,"Duration (seconds)":}.
    """
    results = {}
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
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

def check_log_exists(task_id: str):
    """Check whether the log file exists

    Args:
        task_id (str): task id.

    Returns:
        bool: Does the log file exist.
    """
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
    if os.path.exists(log_path):
        return True
    else:
        return False

def list_to_string(lst: list):
    return " ".join(str(i) for i in lst)
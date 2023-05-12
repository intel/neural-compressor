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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from frontend.fastapi.task_submitter import TaskSubmitter, Task
from frontend.fastapi.utils import (
    get_config,
    get_cluster_info,
    get_cluster_table,
    serialize, deserialize,
    get_res_during_tuning,
    get_baseline_during_tuning,
    check_log_exists,
    list_to_string)
import sqlite3
import os
import uuid
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import json
import socket
from frontend.fastapi.utils import DB_PATH, TASK_LOG_path

app = FastAPI()
task_monitor_port, result_monitor_port = get_config()
serve = TaskSubmitter(task_monitor_port=task_monitor_port, result_monitor_port=result_monitor_port)
db_path = f"{DB_PATH}/task.db"

@app.get("/")
def read_root():
    return {"message": "Welcome to NeuralSolution OaaS!"}

@app.get("/ping")
def ping():
    count = 0
    msg = "Neural Solution is running."
    for port in [serve.task_monitor_port, serve.result_monitor_port]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((serve.inc_serve_ip, port))
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
    return {"status": "Healthy", "msg": msg} if count == 2 else {"status": "Failed", "msg": msg}

@app.get("/cluster")
def get_cluster():
    return get_cluster_info()

@app.get("/clusters")
def get_cluster():
    return HTMLResponse(content=get_cluster_table())

@app.get("/description")
async def get_description():
    with open("../../doc/user_facing_api.json") as f:
        data = json.load(f)
    return data

def list_to_string(lst: list):
    return " ".join(str(i) for i in lst)

@app.post("/task/submit/")
async def submit_task(task: Task):
    msg = "Task submitted successfully"
    status = "successfully"
    tid = "-1"
    # search the current
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        task_id = str(uuid.uuid4()).replace('-','')
        sql = r"insert into task(id, script_url, optimized, arguments, approach, requirements, workers, status)" +\
         r" values ('{}', '{}', {}, '{}', '{}', '{}', {}, 'pending')".format(task_id, task.script_url, task.optimized,
                list_to_string(task.arguments), task.approach, list_to_string(task.requirements), task.workers)
        cursor.execute(sql)
        conn.commit()
        try:
            serve.submit_task(task_id)
        except ConnectionRefusedError:
            msg = "Task Submitted fail! Make sure inc serve runner is running!"
            status = "failed"
        except Exception as e:
            msg = "Task Submitted fail! {}".format(e)
            status = "failed"
        conn.close()
    else:
        msg = "Task Submitted fail! db not found!"
        return {"msg": msg}
    return {"status": status, "task_id": task_id, "msg": msg}

@app.get("/task/{task_id}")
def get_task_by_id(task_id: str):
    res = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    return {"status": res[0], 'optimized_result': deserialize(res[1]) if res[1] else res[1], "result_path": res[2]}

@app.get("/task/")
def get_all_tasks():
    res = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select * from task")
        res = cursor.fetchall()
        cursor.close()
        conn.close()
    return {"message": res}

@app.get("/task/status/{task_id}")
def get_task_status_by_id(task_id: str):
    res = None
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id, ))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    if not res:
        return {"Please check url."}
    elif res[0] == "done":
        return {"status": res[0], 'optimized_result': deserialize(res[1]) if res[1] else res[1], "result_path": res[2]}
    elif res[0] == "pending":
        return {"task pending"}
    else:
        baseline = get_baseline_during_tuning(task_id)
        result = get_res_during_tuning(task_id)
        return {"status": res[0], "baseline": baseline, "message": result}

@app.get("/task/log/{task_id}")
async def read_logs(task_id: str):
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
    if not os.path.exists(log_path):
        return {"error": "Logfile not found."}
    def stream_logs():
        with open(log_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                yield line.encode()
    return StreamingResponse(stream_logs(), media_type="text/plain")

# Real time output log
class LogEventHandler(FileSystemEventHandler):
    def __init__(self, websocket: WebSocket, task_id, last_position):
        super().__init__()
        self.websocket = websocket
        self.task_id = task_id
        self.loop = asyncio.get_event_loop()
        self.last_position = last_position # record last line
        self.queue = asyncio.Queue()
        self.timer = self.loop.create_task(self.send_messages())


    async def send_messages(self):
        while True:
            try:
                messages = []
                while True:
                    message = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    messages.append(message)
            except asyncio.TimeoutError:
                pass

            if messages:
                await self.websocket.send_text("\n".join(messages))

    def on_modified(self, event):
        log_path = "{}/task_{}.txt".format(TASK_LOG_path, self.task_id)
        with open(log_path, "r") as f:
            # Move the file pointer to the last position
            f.seek(self.last_position)
            lines = f.readlines()
            if lines:
                # Record the current position of file pointer
                self.last_position = f.tell()
                for line in lines:
                    self.queue.put_nowait(line.strip())

# start log watcher
def start_log_watcher(websocket, task_id, last_position):
    observer = Observer()
    # watch log/task_{}.txt
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
    observer.schedule(LogEventHandler(websocket, task_id, last_position), log_path, recursive=False)
    observer.start()
    return observer


@app.websocket("/task/screen/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    if not check_log_exists(task_id=task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    await websocket.accept()

    # send the log that has been written
    log_path = "{}/task_{}.txt".format(TASK_LOG_path, task_id)
    last_position = 0
    previous_log = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            previous_log = f.readlines()
            last_position = f.tell()
        # send previous_log
        if previous_log:
            await websocket.send_text("\n".join([message.strip() for message in previous_log]))

    # start log watcher
    observer = start_log_watcher(websocket, task_id, last_position)
    try:
        while True:
            await asyncio.sleep(1)
            # await websocket.receive_text()
    except WebSocketDisconnect:
        observer.stop()
        await observer.join()

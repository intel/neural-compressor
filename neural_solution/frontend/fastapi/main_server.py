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
"""Fast api server."""
import asyncio
import json
import os
import socket
import sqlite3
import uuid
import zipfile

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from neural_solution.config import config
from neural_solution.frontend.task_submitter import Task, task_submitter
from neural_solution.frontend.utility import (
    check_log_exists,
    deserialize,
    get_baseline_during_tuning,
    get_cluster_info,
    get_cluster_table,
    get_res_during_tuning,
    list_to_string,
    serialize,
    is_valid_task,
)
from neural_solution.utils.utility import get_db_path, get_task_log_workspace, get_task_workspace

# Get config from Launcher.sh
task_monitor_port = None
result_monitor_port = None
db_path = None

app = FastAPI()


import argparse

args = None


def parse_arguments():
    """Parse the command line options."""
    parser = argparse.ArgumentParser(description="Frontend with RESTful API")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="The address to submit task.")
    parser.add_argument("-FP", "--fastapi_port", type=int, default=8000, help="Port to submit task by user.")
    parser.add_argument("-TMP", "--task_monitor_port", type=int, default=2222, help="Port to monitor task.")
    parser.add_argument("-RMP", "--result_monitor_port", type=int, default=3333, help="Port to monitor result.")
    parser.add_argument("-WS", "--workspace", type=str, default="./", help="Work space.")
    args = parser.parse_args()
    return args


@app.get("/")
def read_root():
    """Root route."""
    return {"message": "Welcome to Neural Solution!"}


@app.get("/ping")
def ping():
    """Test status of services.

    Returns:
        json: the status of services and message
    """
    count = 0
    msg = "Neural Solution is running."
    for port in [config.task_monitor_port, config.result_monitor_port]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((config.service_address, port))
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
    """Get the cluster info.

    Returns:
        json: the cluster info.
    """
    db_path = get_db_path(config.workspace)
    return get_cluster_info(db_path=db_path)


@app.get("/clusters")
def get_clusters():
    """Get the cluster info.

    Returns:
        HTMLResponse: html table of the cluster info
    """
    db_path = get_db_path(config.workspace)
    return HTMLResponse(content=get_cluster_table(db_path=db_path))


@app.get("/description")
async def get_description():
    """Get user oriented API descriptions.

    Returns:
        json: API descriptions
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "..", "user_facing_api.json")) as f:
        data = json.load(f)
    return data


@app.post("/task/submit/")
async def submit_task(task: Task):
    """Submit task.

    Args:
        task (Task): _description_
        Fields:
            task_id: The task id
            arguments: The task command
            workers: The requested resource unit number
            status: The status of the task: pending/running/done
            result: The result of the task, which is only value-assigned when the task is done

    Returns:
        json: status , id of task and messages.
    """
    if not is_valid_task:
        raise HTTPException(status_code=422, detail="Invalid task")

    msg = "Task submitted successfully"
    status = "successfully"
    # search the current
    db_path = get_db_path(config.workspace)

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
            msg = "Task Submitted fail! Make sure Neural Solution runner is running!"
            status = "failed"
        except Exception as e:
            msg = "Task Submitted fail! {}".format(e)
            status = "failed"
        conn.close()
    else:
        msg = "Task Submitted fail! db not found!"
        return {"msg": msg}  # TODO to align with return message when submit task successfully
    return {"status": status, "task_id": task_id, "msg": msg}


@app.get("/task/{task_id}")
def get_task_by_id(task_id: str):
    """Get task status, result, quantized model path according to id.

    Args:
        task_id (str): the id of task.

    Returns:
        json: task status, result, quantized model path
    """
    res = None
    db_path = get_db_path(config.workspace)
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    return {"status": res[0], "optimized_result": deserialize(res[1]) if res[1] else res[1], "result_path": res[2]}


@app.get("/task/")
def get_all_tasks():
    """Get task table.

    Returns:
        json: task table
    """
    res = None
    db_path = get_db_path(config.workspace)
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select * from task")
        res = cursor.fetchall()
        cursor.close()
        conn.close()
    return {"message": res}


@app.get("/task/status/{task_id}")
def get_task_status_by_id(request: Request, task_id: str):
    """Get task status and information according to id.

    Args:
        task_id (str): the id of task.

    Returns:
        json: task status and information
    """
    status = "unknown"
    tuning_info = {}
    optimization_result = {}

    res = None
    db_path = get_db_path(config.workspace)
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    if not res:
        status = "Please check url."
    elif res[0] == "done":
        status = res[0]
        optimization_result = deserialize(res[1]) if res[1] else res[1]
        download_url = str(request.base_url) + "download/" + task_id
        optimization_result["result_path"] = download_url
    elif res[0] == "pending":
        status = "pending"
    else:
        baseline = get_baseline_during_tuning(task_id, get_task_log_workspace(config.workspace))
        tuning_result = get_res_during_tuning(task_id, get_task_log_workspace(config.workspace))
        status = res[0]
        tuning_info = {"baseline": baseline, "message": tuning_result}
    result = {"status": status, "tuning_info": tuning_info, "optimization_result": optimization_result}
    return result


@app.get("/task/log/{task_id}")
async def read_logs(task_id: str):
    """Get the log of task according to id.

    Args:
        task_id (str): the id of task.

    Returns:
        StreamingResponse: text stream

    Yields:
        str: log lines
    """
    log_path = "{}/task_{}.txt".format(get_task_log_workspace(config.workspace), task_id)
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
    """Responsible for monitoring log changes and sending logs to clients.

    Args:
        FileSystemEventHandler (FileSystemEventHandler): Base file system event handler that overriding methods from.
    """

    def __init__(self, websocket: WebSocket, task_id, last_position):
        """Init.

        Args:
            websocket (WebSocket): websocket connection
            task_id (str): the id of task
            last_position (int): The last line position of the existing log.
        """
        super().__init__()
        self.websocket = websocket
        self.task_id = task_id
        self.loop = asyncio.get_event_loop()
        self.last_position = last_position  # record last line
        self.queue = asyncio.Queue()
        self.timer = self.loop.create_task(self.send_messages())

    async def send_messages(self):
        """Send messages to the client."""
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
        """File modification event."""
        log_path = "{}/task_{}.txt".format(get_task_log_workspace(config.workspace), self.task_id)
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
    """Start log watcher.

    Args:
        websocket (WebSocket): websocket connection
        task_id (str): the id of task.
        last_position (int): The last line position of the existing log.

    Returns:
       Observer : monitor log file changes
    """
    observer = Observer()
    # watch log/task_{}.txt
    log_path = "{}/task_{}.txt".format(get_task_log_workspace(config.workspace), task_id)
    observer.schedule(LogEventHandler(websocket, task_id, last_position), log_path, recursive=False)
    observer.start()
    return observer


@app.websocket("/task/screen/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """Real time log output.

    Args:
        websocket (WebSocket): websocket connection
        task_id (str): the id of task.

    Raises:
        HTTPException: exception
    """
    if not check_log_exists(task_id=task_id, task_log_path=get_task_log_workspace(config.workspace)):
        raise HTTPException(status_code=404, detail="Task not found")
    await websocket.accept()

    # send the log that has been written
    log_path = "{}/task_{}.txt".format(get_task_log_workspace(config.workspace), task_id)
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


@app.get("/download/{task_id}")
async def download_file(task_id: str):
    """Download quantized model.

    Args:
        task_id (str): the task id

    Raises:
        HTTPException: 400, Please check URL
        HTTPException: 404, Task failed, file not found

    Returns:
        FileResponse: quantized model of zip file format
    """
    db_path = get_db_path(config.workspace)
    if os.path.isfile(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(r"select status, result, q_model_path from task where id=?", (task_id,))
        res = cursor.fetchone()
        cursor.close()
        conn.close()
    if res is None:
        raise HTTPException(status_code=400, detail="Please check URL")
    if res[0] != "done":
        raise HTTPException(status_code=404, detail="Task failed, file not found")
    path = res[2]
    zip_filename = "quantized_model.zip"
    zip_filepath = os.path.abspath(os.path.join(get_task_workspace(config.workspace), task_id, zip_filename))
    # create zipfile and add file
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.basename(file_path))

    return FileResponse(
        zip_filepath,
        media_type="application/octet-stream",
        filename=zip_filename,
        background=BackgroundTask(os.remove, zip_filepath),
    )


if __name__ == "__main__":
    # parse the args and modified the config accordingly
    args = parse_arguments()
    config.workspace = args.workspace
    db_path = get_db_path(config.workspace)
    config.task_monitor_port = args.task_monitor_port
    config.result_monitor_port = args.result_monitor_port
    # initialize the task submitter
    task_submitter.task_monitor_port = config.task_monitor_port
    task_submitter.result_monitor_port = config.result_monitor_port
    config.service_address = task_submitter.service_address
    # start the app
    uvicorn.run(app, host=args.host, port=args.fastapi_port)

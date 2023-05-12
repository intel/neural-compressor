import socket
import json
import time

from backend.constant import TASK_MONITOR_PORT, RESULT_MONITOR_PORT
from common import build_task_json
def serialize(request: dict) -> bytes:
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    return json.loads(request)

def send_task():
    s = socket.socket()
    s.connect(("localhost", TASK_MONITOR_PORT))
    s.send(serialize({"task_id": 3, "unit_num": 3, "cmd": "ls -l"}))
    s.close()

send_task()
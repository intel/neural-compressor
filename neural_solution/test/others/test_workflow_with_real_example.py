import socket
import json
import time

from backend.constant import TASK_MONITOR_PORT, RESULT_MONITOR_PORT
from common import build_task_json
def serialize(request: dict) -> bytes:
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    return json.loads(request)

def send_task2():
    s = socket.socket()
    s.connect(("localhost", TASK_MONITOR_PORT))
    task_json = build_task_json()
    s.send(serialize(task_json))
    s.close()
send_task2()
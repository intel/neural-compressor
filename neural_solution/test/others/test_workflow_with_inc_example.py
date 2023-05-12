import socket
import json
import time
import sqlite3
import os

from backend.constant import TASK_MONITOR_PORT, RESULT_MONITOR_PORT
from common import build_task_json
def serialize(request: dict) -> bytes:
    return json.dumps(request).encode()

def deserialize(request: bytes) -> dict:
    return json.loads(request)

def send_task():
    s = socket.socket()
    s.connect(("localhost", TASK_MONITOR_PORT))
    # TODO send the task(example) path to run.sh
    arguments = "bash ./test/test_examples/run_inc_real_task.sh"
    task_json = build_task_json(arguments=arguments)
    s.send(serialize(task_json))
    s.close()

def send_task2():
    # TODO send the task(example) path to run.sh
    arguments = "python -u ./run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval \
--max_seq_length 128 --per_device_eval_batch_size 16 --no_cuda \
--output_dir ./int8_model_dir --performance --overwrite_output_dir"
    url = "https://github.com/intel/neural-compressor/blob/master/examples/pytorch/nlp/huggingface_models/\
text-classification/quantization/qat/fx/run_glue.py"

    conn = sqlite3.connect("../db/task.db")
    cursor = conn.cursor()
    cursor.execute(r"insert into task(arguments, unit_num, script_url) values ('{}', {}, '{}')"
            .format(arguments, 3, url))
    tid = cursor.lastrowid
    cursor.execute(r"select * from task")
    conn.commit()

    task_id, arguments, unit_num, status, script_url, result = cursor.fetchone()
    print(task_id, arguments, unit_num, status, script_url, result)
    task_json = build_task_json(task_id=task_id, arguments=arguments, url=url)
    s = socket.socket()
    s.connect(("localhost", TASK_MONITOR_PORT))
    s.send(serialize(task_json))
    s.close()

def send_task_neural_coder():
    # TODO send the task(example) path to run.sh
    arguments = "--model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result"
    url = "https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification\
/run_glue.py"

    conn = sqlite3.connect("../db/task.db")
    cursor = conn.cursor()
    cursor.execute(r"insert into task(arguments, unit_num, optimized, approach, script_url) values ('{}', {}, {}, '{}', '{}')"
            .format(arguments, 3, False, "static", url))
    tid = cursor.lastrowid
    cursor.execute(r"select * from task order by id desc")
    conn.commit()

    task_id, arguments, unit_num, status, script_url, optimized, approach, q_model_path, result, q_model_path = cursor.fetchone()
    print(task_id, arguments, unit_num, status, script_url, optimized, approach, q_model_path, result, q_model_path)
    task_json = build_task_json(task_id=task_id, arguments=arguments, url=url)
    s = socket.socket()
    s.connect(("localhost", TASK_MONITOR_PORT))
    s.send(serialize(task_json))
    s.close()

# send_task2()
send_task_neural_coder()

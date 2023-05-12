from backend.scheduler import Scheduler
from backend.task_db import TaskDB
from utils.utility import build_local_cluster
from common import build_task

def test_launch_task():
    task_db = TaskDB()
    cluster = build_local_cluster()
    task = build_task()
    scheduler = Scheduler(cluster=cluster, task_db=task_db, result_monitor_port=None)
    resource = cluster.get_free_socket(task.unit_num)
    scheduler.launch_task(task=task, resource=resource)
    
test_launch_task()
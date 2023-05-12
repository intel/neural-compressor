def build_task(task_id = None, arguments="bash ./test/test_examples/run_inc_task.sh", url=None):
    import random
    from backend.task import Task
    if not task_id:
        task_id = random.randint(100, 1000)
    task1 = Task(task_id = task_id,
                 arguments=arguments,
                 unit_num=3,
                 status='pending',
                 script_url=url)
    return task1


def build_task_json(task_id = None, arguments="bash ./test/test_examples/run_inc_task.sh", url=None):
    import datetime
    if not task_id:
        task_id = datetime.datetime.now().timestamp()
    task_json = {"task_id": task_id,
                 "unit_num": 3,
                 "arguments": arguments,
                 "optimized": False,
                 "approach": "static",
                 "script_url": url
                 }
    return task_json


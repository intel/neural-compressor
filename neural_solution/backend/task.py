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

"""Neural Solution task."""
class Task:
    """A Task is an abstraction of a user tuning request that is handled in neural solution service.

    Attributes:
        task_id: The task id
        arguments: The task command
        workers: The requested resource unit number
        status: The status of the task: pending/running/done
        result: The result of the task, which is only value-assigned when the task is done
    """

    def __init__(self, task_id, arguments, workers, status,  script_url, \
        optimized, approach, requirement, result="", q_model_path=""):
        """Init task.

        Args:
            task_id (_type_): the id of task
            arguments (_type_): the running arguments for task.
            workers (_type_): the resources.
            status (_type_): "pending", "running", "done", "failed"
            script_url (_type_): the python script address
            approach (_type_): the quantization method
            requirement (_type_): python packages
            result (str, optional): the result of task. Defaults to "".
            q_model_path (str, optional): the quantized model path. Defaults to "".
        """
        self.task_id = task_id
        self.arguments = arguments
        self.workers = workers
        self.status = status
        self.script_url = script_url
        self.optimized = optimized
        self.approach = approach
        self.requirement = requirement
        self.result = result
        self.q_model_path = q_model_path
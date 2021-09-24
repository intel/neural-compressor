# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
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

"""Tuning history."""
import os.path
from time import sleep
from typing import Optional

from neural_compressor.utils.utility import get_tuning_history
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.web.service.history_snapshot_parser import HistorySnapshotParser


def tuning_history(workload_id: str) -> dict:
    """Get tuning history for requested Workload."""
    from neural_compressor.ux.web.service.workload import WorkloadService

    workload_data = WorkloadService.get_workload_data_by_id(workload_id)

    response = {
        "workload_id": workload_data.get("id"),
    }

    tuning_data = _build_tuning_history(workload_data)
    response.update(tuning_data)

    return response


def _build_tuning_history(workload_data: dict) -> dict:
    """Build tuning history data response."""
    history_path = tuning_history_filename(workload_data.get("id", ""))
    if not os.path.isfile(history_path):
        raise NotFoundException(f"Unable to find tuning history file {history_path}")
    history_snapshot = get_tuning_history(history_path)

    config_path = workload_data.get("config_path")
    if not config_path:
        raise NotFoundException("Unable to find config file")
    config = Config()
    config.load(config_path)
    objective = config.tuning.objective

    history_snapshot_parser = HistorySnapshotParser(
        history_snapshot,
        "performance" == objective,
    )
    tuning_data = history_snapshot_parser.parse_history_snapshot()

    if not tuning_data.get("baseline_accuracy"):
        raise NotFoundException("Can't find baseline accuracy")

    baseline_accuracy = tuning_data.get("baseline_accuracy", 0)
    accuracy_criterion = config.tuning.accuracy_criterion
    if accuracy_criterion.relative:
        accuracy_criterion_type = "relative"
        accuracy_criterion_value = accuracy_criterion.relative
        minimal_accuracy = baseline_accuracy * (1 - accuracy_criterion_value)
    elif accuracy_criterion.absolute:
        accuracy_criterion_type = "absolute"
        accuracy_criterion_value = accuracy_criterion.absolute
        minimal_accuracy = baseline_accuracy - accuracy_criterion_value
    else:
        raise NotFoundException("Unknown accuracy type")

    response = {
        "accuracy_criterion_type": accuracy_criterion_type,
        "accuracy_criterion_value": accuracy_criterion_value,
        "minimal_accuracy": minimal_accuracy,
    }
    response.update(tuning_data)
    return response


def tuning_history_filename(workload_id: str) -> str:
    """Build tuning history filename based on id."""
    from neural_compressor.ux.web.service.workload import WorkloadService

    workload_data = WorkloadService.get_workload_data_by_id(workload_id)
    workload_path = workload_data.get("workload_path", "")
    history_path = os.path.join(workload_path, "history.snapshot")

    return history_path


class Watcher:
    """Tuning history watcher that sends update on file change."""

    def __init__(self, workload_id: str) -> None:
        """Initialize object."""
        self.workload_id = workload_id
        self.watch = False
        self.tuning_history_filename = tuning_history_filename(workload_id)
        self.last_tuning_history_timestamp = self.history_file_modification_time()

    def stop(self) -> None:
        """Signal watcher to stop."""
        self.watch = False

    def __call__(self) -> None:
        """Execute the watch."""
        from neural_compressor.ux.web.service.workload import WorkloadService

        self.watch = True
        while self.watch:
            if self.was_history_file_changed():
                WorkloadService.send_history_snapshot(self.workload_id)
            sleep(10)

    def was_history_file_changed(self) -> bool:
        """Check if history file was changed since last check."""
        current_tuning_history_file_timestamp = self.history_file_modification_time()
        if current_tuning_history_file_timestamp != self.last_tuning_history_timestamp:
            self.last_tuning_history_timestamp = current_tuning_history_file_timestamp
            return True
        return False

    def history_file_modification_time(self) -> Optional[float]:
        """Get modification date of history file."""
        try:
            return os.path.getmtime(self.tuning_history_filename)
        except OSError:
            return None

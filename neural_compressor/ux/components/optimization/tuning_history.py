# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
from typing import List, Optional

from neural_compressor.utils.utility import get_tuning_history
from neural_compressor.ux.components.db_manager.db_operations import OptimizationAPIInterface
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.components.optimization.tune.tuning import AccuracyCriterion, Tuning
from neural_compressor.ux.utils.exceptions import InternalException, NotFoundException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.web.communication import MessageQueue
from neural_compressor.ux.web.service.history_snapshot_parser import HistorySnapshotParser

mq = MessageQueue()


def tuning_history(optimization: Optimization) -> dict:
    """Get tuning history for requested Tuning."""
    response = {
        "optimization_id": optimization.id,
    }
    history_path = tuning_history_path(optimization.workdir)

    tuning_data = _build_tuning_history(optimization, history_path)

    response.update(tuning_data)

    return response


def _build_tuning_history(optimization: Optimization, history_path: str) -> dict:
    """Build tuning history data response."""
    if not os.path.isfile(history_path):
        raise NotFoundException(f"Unable to find tuning history file {history_path}")
    history_snapshot = get_tuning_history(history_path)

    objective: List[str] = ["performance"]
    if isinstance(optimization, Tuning) and optimization.tuning_details.objective:
        objective = [optimization.tuning_details.objective]

    history_snapshot_parser = HistorySnapshotParser(
        history_snapshot,
        "performance" in objective,
    )
    tuning_data = history_snapshot_parser.parse_history_snapshot()

    baseline_accuracy: float = 0.0
    minimal_accuracy: float = 0.0
    accuracy_criterion: AccuracyCriterion = AccuracyCriterion()
    accuracy_criterion.type = "relative"
    accuracy_criterion.threshold = 0.1
    if isinstance(optimization, Tuning):
        if tuning_data.baseline_accuracy is not None:
            baseline_accuracy = tuning_data.baseline_accuracy[0]
        accuracy_criterion = optimization.tuning_details.accuracy_criterion
        if accuracy_criterion.type == "relative":
            minimal_accuracy = baseline_accuracy * (1 - accuracy_criterion.threshold)
        elif accuracy_criterion.type == "absolute":
            minimal_accuracy = baseline_accuracy - accuracy_criterion.threshold
        else:
            raise NotFoundException("Unknown accuracy type")

    response = {
        "accuracy_criterion_type": accuracy_criterion.type,
        "accuracy_criterion_value": accuracy_criterion.threshold,
    }
    tuning_data.minimal_accuracy = minimal_accuracy

    serialized_tuning_data = tuning_data.serialize()

    if not isinstance(serialized_tuning_data, dict):
        raise InternalException("Incorrect type of tuning data.")
    response.update(serialized_tuning_data)
    return response


def tuning_history_path(optimization_workdir: str) -> str:
    """Build path to tuning history filename."""
    return os.path.join(optimization_workdir, "history.snapshot")


class Watcher:
    """Tuning history watcher that sends update on file change."""

    def __init__(self, request_id: str, optimization: Optimization) -> None:
        """Initialize object."""
        self.optimization = optimization
        self.request_id = request_id
        self.watch = False
        self.tuning_history_path = tuning_history_path(optimization.workdir)
        self.last_tuning_history_timestamp = self.history_file_modification_time()

    def stop(self, process_succeeded: bool) -> None:
        """Signal watcher to stop and dump tuning history to database."""
        self.watch = False
        if not process_succeeded:
            log.debug("Tuning process failed. Skipping collecting tuning history.")
            return
        history = tuning_history(self.optimization)
        OptimizationAPIInterface.add_tuning_history(
            optimization_id=self.optimization.id,
            tuning_history=history,
        )

    def __call__(self) -> None:
        """Execute the watch."""
        self.watch = True
        while self.watch:
            if self.was_history_file_changed():
                TuningHistory.send_history_snapshot(self.request_id, self.optimization)
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
            return os.path.getmtime(self.tuning_history_path)
        except OSError:
            return None


class TuningHistory:
    """Tuning history class."""

    @staticmethod
    def send_history_snapshot(request_id: str, optimization: Optimization) -> None:
        """Get tuning history for requested Workload."""
        try:
            response = tuning_history(optimization)
            response.update({"request_id": request_id})
            mq.post_success("tuning_history", response)
        except NotFoundException:
            mq.post_error(
                "tuning_history",
                {
                    "optimization_id": optimization.id,
                },
            )

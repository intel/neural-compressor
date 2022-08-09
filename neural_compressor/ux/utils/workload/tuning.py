# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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
"""Configuration tuning module."""

from typing import Any, Dict, List, Optional, Union

from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.utils import (
    parse_bool_value,
    parse_to_float_list,
    parse_to_string_list,
)


class Strategy(JsonSerializer):
    """Configuration Strategy class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize configuration Strategy class."""
        super().__init__()
        # [Required] One of neural_compressor.strategy.STRATEGIES
        self.name: str = data.get("name", "basic")
        self.sigopt_api_token: Optional[str] = data.get("sigopt_api_token", None)

        self.accuracy_weight: Optional[float] = data.get("accuracy_weight", None)
        self.latency_weight: Optional[float] = data.get("latency_weight", None)


class MultiObjectives(JsonSerializer):
    """Configuration MultiObjectives class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize configuration MultiObjectives class."""
        super().__init__()
        self._objective: List[str] = data.get("objective", [])
        self._weight: List[float] = data.get("weight", [])

    @property
    def objective(self) -> List[str]:
        """Get objectives."""
        return self._objective

    @objective.setter
    def objective(self, value: Union[None, str, List[str]]) -> None:
        """Set inputs value."""
        self._objective = parse_to_string_list(value)

    @property
    def weight(self) -> List[float]:
        """Get weights."""
        return self._weight

    @weight.setter
    def weight(self, value: Union[None, float, List[float]]) -> None:
        """Set weights value."""
        self._weight = parse_to_float_list(value)


class AccCriterion(JsonSerializer):
    """Configuration AccCriterion class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize configuration AccCriterion class."""
        super().__init__()
        self.relative: Optional[float] = data.get(
            "relative",
            None,
        )  # [Optional] (INT8-FP32)/FP32
        self.absolute: Optional[float] = data.get(
            "absolute",
            None,
        )  # [Optional] INT8-FP32

        # Set default accuracy criterion to relative
        if self.relative is None and self.absolute is None:
            self.relative = 0.1


class ExitPolicy(JsonSerializer):
    """Configuration ExitPolicy class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration ExitPolicy class."""
        super().__init__()
        self.timeout: Optional[int] = data.get("timeout", None)

        self.max_trials: Optional[int] = data.get("max_trials", None)

        self.performance_only: Optional[bool] = data.get("performance_only", None)


class Workspace(JsonSerializer):
    """Configuration Workspace class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Workspace class."""
        super().__init__()
        self.path: Optional[str] = data.get("path", None)  # [Optional]

        self.resume: Optional[str] = data.get("resume", None)  # [Optional]


class Diagnosis(JsonSerializer):
    """Configuration Diagnosis class."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Configuration Diagnosis class."""
        super().__init__()
        if data is None:
            data = {}
        self.diagnosis_after_tuning: bool = data.get("diagnosis_after_tuning", True)
        self.op_list: Optional[List[str]] = data.get("op_list", None)
        self.iteration_list: Optional[List[int]] = data.get("iteration_list", None)
        self.inspect_type: Optional[str] = data.get("inspect_type", None)
        self.save_to_disk: bool = data.get("save_to_disk", True)
        self.save_path: Optional[str] = data.get("save_path", None)


class Tuning(JsonSerializer):
    """Configuration Tuning class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Tuning class."""
        super().__init__()
        self.strategy: Strategy = Strategy()
        if data.get("strategy"):
            self.strategy = Strategy(data.get("strategy", {}))

        self.accuracy_criterion: AccCriterion = AccCriterion(
            data.get("accuracy_criterion", {}),
        )

        self.multi_objectives: Optional[MultiObjectives] = None
        if data.get("multi_objectives"):
            self.multi_objectives = MultiObjectives(data.get("multi_objectives", {}))

        self.exit_policy: Optional[ExitPolicy] = None
        if data.get("exit_policy"):
            self.exit_policy = ExitPolicy(data.get("exit_policy", {}))

        self.random_seed: Optional[int] = data.get("random_seed", None)

        self.tensorboard: Optional[bool] = data.get("tensorboard", None)

        self.workspace: Optional[Workspace] = None
        if data.get("workspace", {}):
            self.workspace = Workspace(data.get("workspace", {}))

        self.diagnosis: Optional[Diagnosis] = None
        if data.get("diagnosis", {}):
            self.diagnosis = Diagnosis(data.get("diagnosis", {}))

    def set_timeout(self, timeout: int) -> None:
        """Update tuning timeout in config."""
        try:
            timeout = int(timeout)
            if timeout < 0:
                raise ValueError
        except ValueError:
            raise ClientErrorException(
                "The timeout value is not valid. " "Timeout should be non negative integer.",
            )
        if self.exit_policy:
            self.exit_policy.timeout = timeout
        else:
            self.exit_policy = ExitPolicy({"timeout": timeout})

    def set_max_trials(self, max_trials: int) -> None:
        """Update max tuning trials in config."""
        try:
            max_trials = int(max_trials)
            if max_trials < 0:
                raise ValueError
        except ValueError:
            raise ClientErrorException(
                "The max trials value is not valid. " "Max trials should be non negative integer.",
            )
        if self.exit_policy:
            self.exit_policy.max_trials = max_trials
        else:
            self.exit_policy = ExitPolicy({"max_trials": max_trials})

    def set_performance_only(self, performance_only: Any) -> None:
        """Update performance only flag in config."""
        try:
            performance_only = parse_bool_value(performance_only)
        except ValueError:
            raise ClientErrorException(
                "The performance_only flag value is not valid. "
                "Performance_ony should be a boolean.",
            )
        if self.exit_policy:
            self.exit_policy.performance_only = performance_only
        else:
            self.exit_policy = ExitPolicy({"performance_only": performance_only})

    def set_random_seed(self, random_seed: int) -> None:
        """Update random seed value in config."""
        try:
            random_seed = int(random_seed)
        except ValueError:
            raise ClientErrorException(
                "The random seed value is not valid. " "Random seed should be an integer.",
            )
        self.random_seed = random_seed

    def set_workspace(self, path: str) -> None:
        """Update tuning workspace path in config."""
        if self.workspace is None:
            self.workspace = Workspace()
        self.workspace.path = path

        if self.diagnosis is not None:
            self.diagnosis.save_path = path

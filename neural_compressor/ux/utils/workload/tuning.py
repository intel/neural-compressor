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
"""Configuration tuning module."""

from typing import Any, Dict, Optional

from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.utils import parse_bool_value


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


class Tuning(JsonSerializer):
    """Configuration Tuning class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Tuning class."""
        super().__init__()
        self.strategy: Strategy = Strategy()  # [Optional]
        if data.get("strategy"):
            self.strategy = Strategy(data.get("strategy", {}))

        self.accuracy_criterion: AccCriterion = AccCriterion(
            data.get("accuracy_criterion", {}),
        )

        # [Optional] One of neural_compressor.objective.OBJECTIVES
        self.objective: Optional[str] = data.get("objective", None)

        self.exit_policy: Optional[ExitPolicy] = None  # [Optional]
        if data.get("exit_policy"):
            self.exit_policy = ExitPolicy(data.get("exit_policy", {}))

        self.random_seed: Optional[int] = data.get("random_seed", None)

        self.tensorboard: Optional[bool] = data.get("tensorboard", None)

        self.workspace: Workspace = Workspace(data.get("workspace", {}))

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

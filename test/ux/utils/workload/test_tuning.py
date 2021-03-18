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
"""Tuning config test."""

import unittest

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.workload.tuning import (
    AccCriterion,
    ExitPolicy,
    Strategy,
    Tuning,
    Workspace,
)


class TestStrategyConfig(unittest.TestCase):
    """Strategy config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Strategy config test constructor."""
        super().__init__(*args, **kwargs)

    def test_strategy_constructor(self) -> None:
        """Test Strategy config constructor."""
        data = {
            "name": "mse",
            "accuracy_weight": 0.5,
            "latency_weight": 1.0,
        }
        strategy = Strategy(data)

        self.assertEqual(strategy.name, "mse")
        self.assertEqual(strategy.accuracy_weight, 0.5)
        self.assertEqual(strategy.latency_weight, 1.0)

    def test_strategy_constructor_defaults(self) -> None:
        """Test Strategy config constructor defaults."""
        strategy = Strategy()

        self.assertEqual(strategy.name, "basic")
        self.assertIsNone(strategy.accuracy_weight)
        self.assertIsNone(strategy.latency_weight)

    def test_strategy_serializer(self) -> None:
        """Test Strategy config serializer."""
        data = {
            "name": "bayesian",
            "accuracy_weight": 0.5,
            "latency_weight": 1.0,
        }
        strategy = Strategy(data)
        result = strategy.serialize()

        self.assertDictEqual(
            result,
            {
                "name": "bayesian",
                "accuracy_weight": 0.5,
                "latency_weight": 1.0,
            },
        )


class TestAccCriterionConfig(unittest.TestCase):
    """AccCriterion config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Acc Criterion config test constructor."""
        super().__init__(*args, **kwargs)

    def test_acc_critetion_constructor(self) -> None:
        """Test AccCriterion config constructor."""
        data = {
            "relative": 0.01,
            "absolute": 0.02,
        }
        acc_critetion = AccCriterion(data)

        self.assertEqual(acc_critetion.relative, 0.01)
        self.assertEqual(acc_critetion.absolute, 0.02)

    def test_acc_critetion_constructor_defaults(self) -> None:
        """Test AccCriterion config constructor defaults."""
        acc_critetion = AccCriterion()

        self.assertIsNone(acc_critetion.relative)
        self.assertIsNone(acc_critetion.absolute)

    def test_acc_critetion_serializer(self) -> None:
        """Test AccCriterion config serializer."""
        data = {
            "relative": 0.01,
            "absolute": 0.02,
        }
        acc_critetion = AccCriterion(data)
        result = acc_critetion.serialize()

        self.assertDictEqual(
            result,
            {
                "relative": 0.01,
                "absolute": 0.02,
            },
        )


class TestExitPolicyConfig(unittest.TestCase):
    """ExitPolicy config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Exit Policy config test constructor."""
        super().__init__(*args, **kwargs)

    def test_exit_policy_constructor(self) -> None:
        """Test ExitPolicy config constructor."""
        data = {
            "timeout": 60,
            "max_trials": 200,
        }
        exit_policy = ExitPolicy(data)

        self.assertEqual(exit_policy.timeout, 60)
        self.assertEqual(exit_policy.max_trials, 200)

    def test_exit_policy_constructor_defaults(self) -> None:
        """Test ExitPolicy config constructor defaults."""
        exit_policy = ExitPolicy()

        self.assertIsNone(exit_policy.timeout)
        self.assertIsNone(exit_policy.max_trials)

    def test_exit_policy_serializer(self) -> None:
        """Test ExitPolicy config serializer."""
        data = {
            "timeout": 60,
            "max_trials": 200,
        }
        exit_policy = ExitPolicy(data)
        result = exit_policy.serialize()

        self.assertDictEqual(
            result,
            {
                "timeout": 60,
                "max_trials": 200,
            },
        )


class TestWorkspaceConfig(unittest.TestCase):
    """Workspace config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Workspace config test constructor."""
        super().__init__(*args, **kwargs)

    def test_exit_policy_constructor(self) -> None:
        """Test Workspace config constructor."""
        data = {
            "path": "/path/to/workspace",
            "resume": "/path/to/snapshot/file",
        }
        workspace = Workspace(data)

        self.assertEqual(workspace.path, "/path/to/workspace")
        self.assertEqual(workspace.resume, "/path/to/snapshot/file")

    def test_workspace_constructor_defaults(self) -> None:
        """Test Workspace config constructor defaults."""
        workspace = Workspace()

        self.assertIsNone(workspace.path)
        self.assertIsNone(workspace.resume)

    def test_workspace_serializer(self) -> None:
        """Test Workspace config serializer."""
        data = {
            "path": "/path/to/workspace",
            "resume": "/path/to/snapshot/file",
            "additional_field": 1,
        }
        workspace = Workspace(data)
        result = workspace.serialize()

        self.assertDictEqual(
            result,
            {
                "path": "/path/to/workspace",
                "resume": "/path/to/snapshot/file",
            },
        )


class TestTuningConfig(unittest.TestCase):
    """Tuning config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Tuning config test constructor."""
        super().__init__(*args, **kwargs)

    def test_tuning_constructor(self) -> None:
        """Test Tuning config constructor."""
        data = {
            "strategy": {
                "name": "mse",
                "accuracy_weight": 0.5,
                "latency_weight": 1.0,
            },
            "accuracy_criterion": {
                "relative": 0.01,
                "absolute": 0.02,
            },
            "objective": "performance",
            "exit_policy": {
                "timeout": 60,
                "max_trials": 200,
            },
            "random_seed": 12345,
            "tensorboard": True,
            "workspace": {
                "path": "/path/to/workspace",
                "resume": "/path/to/snapshot/file",
            },
        }
        tuning = Tuning(data)

        self.assertIsNotNone(tuning.strategy)
        self.assertEqual(tuning.strategy.name, "mse")
        self.assertEqual(tuning.strategy.accuracy_weight, 0.5)
        self.assertEqual(tuning.strategy.latency_weight, 1.0)

        self.assertIsNotNone(tuning.accuracy_criterion)
        self.assertEqual(tuning.accuracy_criterion.relative, 0.01)
        self.assertEqual(tuning.accuracy_criterion.absolute, 0.02)

        self.assertEqual(tuning.objective, "performance")

        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.timeout, 60)
        self.assertEqual(tuning.exit_policy.max_trials, 200)

        self.assertEqual(tuning.random_seed, 12345)

        self.assertTrue(tuning.tensorboard)

        self.assertIsNotNone(tuning.workspace)
        self.assertEqual(tuning.workspace.path, "/path/to/workspace")
        self.assertEqual(tuning.workspace.resume, "/path/to/snapshot/file")

    def test_tuning_constructor_defaults(self) -> None:
        """Test Tuning config constructor defaults."""
        tuning = Tuning()

        self.assertIsNotNone(tuning.strategy)
        self.assertEqual(tuning.strategy.name, "basic")
        self.assertIsNone(tuning.strategy.accuracy_weight)
        self.assertIsNone(tuning.strategy.latency_weight)

        self.assertIsNotNone(tuning.accuracy_criterion)
        self.assertIsNone(tuning.accuracy_criterion.relative)
        self.assertIsNone(tuning.accuracy_criterion.absolute)

        self.assertIsNone(tuning.objective)
        self.assertIsNone(tuning.exit_policy)
        self.assertIsNone(tuning.random_seed)
        self.assertIsNone(tuning.tensorboard)

        self.assertIsNotNone(tuning.workspace)
        self.assertIsNone(tuning.workspace.path)
        self.assertIsNone(tuning.workspace.resume)

    def test_set_timeout(self) -> None:
        """Test setting timeout in Tuning config."""
        tuning = Tuning()
        tuning.set_timeout(10)
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.timeout, 10)

    def test_set_timeout_with_exit_policy(self) -> None:
        """Test overwriting timeout in Tuning config."""
        tuning = Tuning(
            {
                "exit_policy": {
                    "timeout": 60,
                },
            },
        )
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.timeout, 60)

        tuning.set_timeout(10)
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.timeout, 10)

    def test_set_timeout_from_string(self) -> None:
        """Test overwriting timeout in Tuning config."""
        tuning = Tuning()
        tuning.set_timeout("10")
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.timeout, 10)

    def test_set_timeout_negative(self) -> None:
        """Test overwriting timeout in Tuning config."""
        tuning = Tuning()
        with self.assertRaises(ClientErrorException):
            tuning.set_timeout(-1)

    def test_set_timeout_invalid_string(self) -> None:
        """Test overwriting timeout in Tuning config."""
        tuning = Tuning()
        with self.assertRaises(ClientErrorException):
            tuning.set_timeout("abc")

    def test_set_max_trials(self) -> None:
        """Test setting max_trials in Tuning config."""
        tuning = Tuning()
        tuning.set_max_trials(10)
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.max_trials, 10)

    def test_set_max_trials_with_exit_policy(self) -> None:
        """Test overwriting max_trials in Tuning config."""
        tuning = Tuning(
            {
                "exit_policy": {
                    "max_trials": 60,
                },
            },
        )
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.max_trials, 60)

        tuning.set_max_trials(10)
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.max_trials, 10)

    def test_set_max_trials_from_string(self) -> None:
        """Test overwriting max_trials in Tuning config."""
        tuning = Tuning()
        tuning.set_max_trials("10")
        self.assertIsNotNone(tuning.exit_policy)
        self.assertEqual(tuning.exit_policy.max_trials, 10)

    def test_set_max_trials_negative(self) -> None:
        """Test overwriting max_trials in Tuning config."""
        tuning = Tuning()
        with self.assertRaises(ClientErrorException):
            tuning.set_max_trials(-1)

    def test_set_max_trials_invalid_string(self) -> None:
        """Test overwriting max_trials in Tuning config."""
        tuning = Tuning()
        with self.assertRaises(ClientErrorException):
            tuning.set_max_trials("abc")

    def test_set_random_seed(self) -> None:
        """Test setting random_seed in Tuning config."""
        tuning = Tuning()
        tuning.set_random_seed(123456)
        self.assertEqual(tuning.random_seed, 123456)

    def test_set_random_seed_from_string(self) -> None:
        """Test setting random_seed from string in Tuning config."""
        tuning = Tuning()
        tuning.set_random_seed("123456")
        self.assertEqual(tuning.random_seed, 123456)

    def test_set_random_seed_invalid_string(self) -> None:
        """Test setting random_seed from invalid string in Tuning config."""
        tuning = Tuning()
        with self.assertRaises(ClientErrorException):
            tuning.set_random_seed("abc")

    def test_tuning_serializer(self) -> None:
        """Test Tuning config serializer."""
        data = {
            "strategy": {
                "name": "mse",
                "accuracy_weight": 0.5,
                "latency_weight": 1.0,
            },
            "accuracy_criterion": {
                "relative": 0.01,
                "absolute": 0.02,
            },
            "objective": "performance",
            "exit_policy": {
                "timeout": 60,
                "max_trials": 200,
            },
            "random_seed": 12345,
            "tensorboard": True,
            "workspace": {
                "path": "/path/to/workspace",
                "resume": "/path/to/snapshot/file",
            },
            "additional_field": {"key": "val"},
        }
        tuning = Tuning(data)
        result = tuning.serialize()

        self.assertDictEqual(
            result,
            {
                "strategy": {
                    "name": "mse",
                    "accuracy_weight": 0.5,
                    "latency_weight": 1.0,
                },
                "accuracy_criterion": {
                    "relative": 0.01,
                    "absolute": 0.02,
                },
                "objective": "performance",
                "exit_policy": {
                    "timeout": 60,
                    "max_trials": 200,
                },
                "random_seed": 12345,
                "tensorboard": True,
                "workspace": {
                    "path": "/path/to/workspace",
                    "resume": "/path/to/snapshot/file",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()

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
"""Pruning config test."""

import unittest

from lpot.ux.utils.workload.pruning import Magnitude, Pruning


class TestMagnitudeConfig(unittest.TestCase):
    """Magnitude config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Pruning config constructor."""
        super().__init__(*args, **kwargs)

    def test_magnitude_constructor(self) -> None:
        """Test Magnitude config constructor."""
        data = {
            "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
            "method": "per_channel",
            "init_sparsity": 0.3,
            "target_sparsity": 0.5,
            "start_epoch": 1,
            "end_epoch": 3,
        }
        magnitude = Magnitude(data)

        self.assertEqual(
            magnitude.weights,
            ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
        )
        self.assertEqual(magnitude.method, "per_channel")
        self.assertEqual(magnitude.init_sparsity, 0.3)
        self.assertEqual(magnitude.target_sparsity, 0.5)
        self.assertEqual(magnitude.start_epoch, 1)
        self.assertEqual(magnitude.end_epoch, 3)

    def test_magnitude_constructor_defaults(self) -> None:
        """Test Magnitude config constructor defaults."""
        magnitude = Magnitude()

        self.assertIsNone(magnitude.weights)
        self.assertIsNone(magnitude.method)
        self.assertIsNone(magnitude.init_sparsity)
        self.assertIsNone(magnitude.target_sparsity)
        self.assertIsNone(magnitude.start_epoch)
        self.assertIsNone(magnitude.end_epoch)

    def test_magnitude_serializer_defaults(self) -> None:
        """Test Magnitude config constructor defaults."""
        magnitude = Magnitude()
        result = magnitude.serialize()

        self.assertEqual(type(result), dict)
        self.assertDictEqual(result, {})

    def test_magnitude_serializer(self) -> None:
        """Test Magnitude config constructor."""
        data = {
            "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
            "method": "per_channel",
            "init_sparsity": 0.3,
            "target_sparsity": 0.5,
            "start_epoch": 1,
            "end_epoch": 3,
        }
        magnitude = Magnitude(data)
        result = magnitude.serialize()

        self.assertDictEqual(
            result,
            {
                "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
                "method": "per_channel",
                "init_sparsity": 0.3,
                "target_sparsity": 0.5,
                "start_epoch": 1,
                "end_epoch": 3,
            },
        )


class TestPruningConfig(unittest.TestCase):
    """Pruning config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Pruning config constructor."""
        super().__init__(*args, **kwargs)

    def test_pruning_constructor(self) -> None:
        """Test Pruning config constructor."""
        data = {
            "magnitude": {
                "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
                "method": "per_channel",
                "init_sparsity": 0.3,
                "target_sparsity": 0.5,
                "start_epoch": 1,
                "end_epoch": 3,
            },
            "start_epoch": 0,
            "end_epoch": 2,
            "frequency": 0.5,
            "init_sparsity": 0.25,
            "target_sparsity": 0.75,
        }
        pruning = Pruning(data)

        self.assertEqual(
            pruning.magnitude.weights,
            ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
        )
        self.assertEqual(pruning.magnitude.method, "per_channel")
        self.assertEqual(pruning.magnitude.init_sparsity, 0.3)
        self.assertEqual(pruning.magnitude.target_sparsity, 0.5)
        self.assertEqual(pruning.magnitude.start_epoch, 1)
        self.assertEqual(pruning.magnitude.end_epoch, 3)
        self.assertEqual(pruning.start_epoch, 0)
        self.assertEqual(pruning.end_epoch, 2)
        self.assertEqual(pruning.frequency, 0.5)
        self.assertEqual(pruning.init_sparsity, 0.25)
        self.assertEqual(pruning.target_sparsity, 0.75)

    def test_pruning_constructor_defaults(self) -> None:
        """Test Pruning config constructor defaults."""
        pruning = Pruning()

        self.assertIsNone(pruning.magnitude.weights)
        self.assertIsNone(pruning.magnitude.method)
        self.assertIsNone(pruning.magnitude.init_sparsity)
        self.assertIsNone(pruning.magnitude.target_sparsity)
        self.assertIsNone(pruning.magnitude.start_epoch)
        self.assertIsNone(pruning.magnitude.end_epoch)
        self.assertIsNone(pruning.start_epoch)
        self.assertIsNone(pruning.end_epoch)
        self.assertIsNone(pruning.frequency)
        self.assertIsNone(pruning.init_sparsity)
        self.assertIsNone(pruning.target_sparsity)

    def test_pruning_serializer_defaults(self) -> None:
        """Test Pruning config constructor defaults."""
        pruning = Pruning()
        result = pruning.serialize()

        self.assertDictEqual(result, {})

    def test_pruning_serializer(self) -> None:
        """Test Pruning config constructor."""
        data = {
            "magnitude": {
                "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
                "method": "per_channel",
                "init_sparsity": 0.3,
                "target_sparsity": 0.5,
                "start_epoch": 1,
                "end_epoch": 3,
            },
            "start_epoch": 0,
            "end_epoch": 2,
            "frequency": 0.5,
            "init_sparsity": 0.25,
            "target_sparsity": 0.75,
        }
        pruning = Pruning(data)
        result = pruning.serialize()

        self.assertDictEqual(
            result,
            {
                "magnitude": {
                    "weights": ["layer1.0.conv1.weight", "layer1.0.conv2.weight"],
                    "method": "per_channel",
                    "init_sparsity": 0.3,
                    "target_sparsity": 0.5,
                    "start_epoch": 1,
                    "end_epoch": 3,
                },
                "start_epoch": 0,
                "end_epoch": 2,
                "frequency": 0.5,
                "init_sparsity": 0.25,
                "target_sparsity": 0.75,
            },
        )


if __name__ == "__main__":
    unittest.main()

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
"""Saving Workload test."""
import unittest
from collections import OrderedDict
from typing import List

from neural_compressor.ux.components.configuration_wizard.save_workload import (
    change_performance_dataloader_to_dummy_if_possible,
    get_height_width_from_size,
    get_shape_from_transforms,
)
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.dataloader import Dataset, Transform


class TestUpdateConfigWithDummy(unittest.TestCase):
    """Test updating config with Dummy dataloader."""

    def test_get_shape_from_transforms_for_empty_list(self) -> None:
        """Test getting shape from transforms."""
        transforms: List[Transform] = []
        with self.assertRaisesRegex(NotFoundException, "Unable to detect shape for Dummy dataset"):
            get_shape_from_transforms(transforms)

    def test_get_shape_from_transforms_for_list_of_unknown_transforms(self) -> None:
        """Test getting shape from transforms."""
        transforms = [
            Transform("foo"),
            Transform("bar"),
            Transform("baz"),
        ]
        with self.assertRaisesRegex(NotFoundException, "Unable to detect shape for Dummy dataset"):
            get_shape_from_transforms(transforms)

    def test_get_shape_from_transforms_with_size_parameter(self) -> None:
        """Test getting shape from transforms."""
        for transform_name in [
            "Resize",
            "CenterCrop",
            "RandomResizedCrop",
            "RandomCrop",
            "CropResize",
        ]:
            transforms = [
                Transform("foo"),
                Transform(transform_name, {"size": [10, 20]}),
                Transform("baz"),
            ]
            expected = [10, 20, 3]
            actual = get_shape_from_transforms(transforms)
            self.assertEqual(expected, actual, f"Incorrect shape for {transform_name}")

    def test_get_shape_from_transforms_with_height_and_width_parameters(self) -> None:
        """Test getting shape from transforms."""
        for transform_name in [
            "ResizeCropImagenet",
            "BilinearImagenet",
        ]:
            transforms = [
                Transform("foo"),
                Transform(transform_name, {"height": 10, "width": 20}),
                Transform("baz"),
            ]
            expected = [10, 20, 3]
            actual = get_shape_from_transforms(transforms)
            self.assertEqual(expected, actual, f"Incorrect shape for {transform_name}")

    def test_get_shape_from_CropToBoundingBox(self) -> None:
        """Test getting shape from transforms."""
        transforms = [
            Transform("foo"),
            Transform("CropToBoundingBox", {"target_height": 10, "target_width": 20}),
            Transform("baz"),
        ]
        expected = [10, 20, 3]
        actual = get_shape_from_transforms(transforms)
        self.assertEqual(expected, actual)

    def test_get_shape_from_transforms_with_transposing(self) -> None:
        """Test getting shape from transforms."""
        transforms = [
            Transform("foo"),
            Transform("Resize", {"size": [10, 20]}),
            Transform("baz"),
        ]
        expected = [10, 20, 3]
        actual = get_shape_from_transforms(transforms)
        self.assertEqual(expected, actual, "Incorrect initial shape before transposing")

        rotate_right_transpose = Transform("Transpose", {"perm": [2, 0, 1]})

        # add first transpose
        transforms.append(rotate_right_transpose)
        expected = [3, 10, 20]
        actual = get_shape_from_transforms(transforms)
        self.assertEqual(expected, actual, "Incorrect shape after first transpose")

        # add second transpose
        transforms.append(rotate_right_transpose)
        expected = [20, 3, 10]
        actual = get_shape_from_transforms(transforms)
        self.assertEqual(expected, actual, "Incorrect shape after second transpose")

        # add third transpose
        transforms.append(rotate_right_transpose)
        expected = [10, 20, 3]
        actual = get_shape_from_transforms(transforms)
        self.assertEqual(expected, actual, "Incorrect shape after third transpose")

    def test_get_shape_fails_on_incorrect_transpose(self) -> None:
        """Test getting shape from transforms."""
        transforms = [
            Transform("foo"),
            Transform("Transpose"),
            Transform("baz"),
        ]
        with self.assertRaisesRegex(ValueError, "Unknown value of 'perm' argument in Transpose"):
            get_shape_from_transforms(transforms)

    def test_get_shape_from_different_resize_params(self) -> None:
        """Test getting shape from transforms."""
        sizes = {
            "int": {
                "size": 20,
                "expected": (20, 20),
            },
            "list - one element": {
                "size": [10],
                "expected": (10, 10),
            },
            "list - two elements": {
                "size": [10, 20],
                "expected": (10, 20),
            },
            "list - many elements": {
                "size": [10, 20, 30],
                "expected": (None, None),
            },
            "other": {
                "size": {"foo": "bar"},
                "expected": (None, None),
            },
        }

        for size_name, size_definition in sizes.items():
            expected = size_definition["expected"]  # type: ignore
            actual = get_height_width_from_size(size_definition["size"])  # type: ignore
            self.assertEqual(expected, actual, f"Incorrect size detection for {size_name}")

    def test_does_nothing_when_incorrect_domain(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("NLP", config)
        actual = config.serialize()

        self.assertEqual(expected, actual, "Config should not be changed by not supported domain")

    def test_does_nothing_when_evaluation_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.evaluation = None
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_evaluation_performance_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.evaluation.performance = None  # type: ignore  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_evaluation_performance_dataloader_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.evaluation.performance.dataloader = None  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_evaluation_performance_dataloader_dataset_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.evaluation.performance.dataloader.dataset = None  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_quantization_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.quantization = None
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_quantization_calibration_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.quantization.calibration = None  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_quantization_calibration_dataloader_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.quantization.calibration.dataloader = None  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_quantization_calibration_dataloader_transform_empty(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.quantization.calibration.dataloader.transform.clear()  # type: ignore
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_does_nothing_when_unable_to_guess_shapes(
        self,
    ) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        config.quantization.calibration.dataloader.transform = OrderedDict(  # type: ignore
            {"foo": Transform("foo")},
        )
        expected = config.serialize()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)
        actual = config.serialize()

        self.assertEqual(expected, actual)

    def test_change_performance_dataloader_to_dummy(self) -> None:
        """Test change_performance_dataloader_to_dummy_if_possible."""
        config = self.get_eligible_config()

        change_performance_dataloader_to_dummy_if_possible("image_recognition", config)

        self.assertEqual(
            OrderedDict(),
            config.evaluation.performance.dataloader.transform,  # type: ignore
        )

        expected_dataset = Dataset(
            "dummy_v2",
            {
                "input_shape": [10, 20, 3],
                "label_shape": [1],
            },
        )
        self.assertEqual(
            expected_dataset,
            config.evaluation.performance.dataloader.dataset,  # type: ignore
        )

    def get_eligible_config(self) -> Config:
        """Build a Config that should be updated to Dummy."""
        return Config(
            {
                "evaluation": {
                    "performance": {
                        "dataloader": {
                            "dataset": {
                                "foo": {},
                            },
                        },
                    },
                },
                "quantization": {
                    "calibration": {
                        "dataloader": {
                            "dataset": {
                                "foo": {},
                            },
                            "transform": {
                                "Resize": {
                                    "size": [10, 20],
                                },
                            },
                        },
                    },
                },
            },
        )


if __name__ == "__main__":
    unittest.main()

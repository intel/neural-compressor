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
"""Dataloader config test."""

import unittest

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.workload.dataloader import Dataloader, Dataset, Filter, LabelBalance, Transform


class TestDatasetConfig(unittest.TestCase):
    """Dataset config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize dataset config test case."""
        super().__init__(*args, **kwargs)

    def test_dataset_constructor(self) -> None:
        """Test Dataset config constructor."""
        name = "TestDataset"
        data = {
            "dataset_param": "/some/path",
            "bool_param": True,
            "list_param": ["item1", "item2"],
        }
        dataset = Dataset(name, data)

        self.assertEqual(dataset.name, "TestDataset")
        self.assertDictEqual(
            dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )

    def test_dataset_constructor_without_params(self) -> None:
        """Test Dataset config constructor without parameters."""
        dataset = Dataset("TestDataset")

        self.assertEqual(dataset.name, "TestDataset")
        self.assertDictEqual(dataset.params, {})

    def test_dataser_serializer_without_params(self) -> None:
        """Test Dataset config serializer without parameters."""
        dataset = Dataset("TestDataset")
        result = dataset.serialize()

        self.assertEqual(type(result), dict)
        self.assertDictEqual(result, {"TestDataset": None})

    def test_dataset_serializer(self) -> None:
        """Test Dataset config serializer."""
        name = "TestDataset"
        data = {
            "dataset_param": "/some/path",
            "bool_param": True,
            "list_param": ["item1", "item2"],
        }
        dataset = Dataset(name, data)
        result = dataset.serialize()

        self.assertDictEqual(
            result,
            {
                "TestDataset": {
                    "dataset_param": "/some/path",
                    "bool_param": True,
                    "list_param": ["item1", "item2"],
                },
            },
        )


class TestLabelBalanceConfig(unittest.TestCase):
    """LabelBalance config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize label balance config test case."""
        super().__init__(*args, **kwargs)

    def test_label_balance_constructor(self) -> None:
        """Test label balance config constructor."""
        data = {"size": 1, "redundant_key": "val"}
        label_balance = LabelBalance(data)

        self.assertEqual(label_balance.size, 1)

    def test_label_balance_constructor_defaults(self) -> None:
        """Test label balance config constructor defaults."""
        label_balance = LabelBalance()

        self.assertIsNone(label_balance.size)

    def test_label_balance_serializer(self) -> None:
        """Test label balance config serializer."""
        data = {"size": 1, "redundant_key": "val"}
        label_balance = LabelBalance(data)

        self.assertDictEqual(
            label_balance.serialize(),
            {
                "size": 1,
            },
        )

    def test_label_balance_serializer_defaults(self) -> None:
        """Test label balance config serializer defaults."""
        label_balance = LabelBalance()

        self.assertDictEqual(
            label_balance.serialize(),
            {},
        )


class TestFilterConfig(unittest.TestCase):
    """Filter config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize filter config test case."""
        super().__init__(*args, **kwargs)

    def test_filter_constructor(self) -> None:
        """Test filter config constructor."""
        data = {
            "LabelBalance": {
                "size": 1,
            },
        }
        filter_config = Filter(data)

        self.assertIsNotNone(filter_config.LabelBalance)
        self.assertEqual(type(filter_config.LabelBalance), LabelBalance)
        self.assertEqual(filter_config.LabelBalance.size, 1)

    def test_filter_serializer(self) -> None:
        """Test filter config serializer."""
        data = {
            "LabelBalance": {
                "size": 1,
            },
            "redundant_key": {"some_key": "value"},
        }
        filter_config = Filter(data)

        self.assertDictEqual(
            filter_config.serialize(),
            {"LabelBalance": {"size": 1}},
        )

    def test_filter_constructor_defaults(self) -> None:
        """Test filter config constructor defaults."""
        filter_config = Filter()

        self.assertIsNone(filter_config.LabelBalance)

    def test_filter_serializer_defaults(self) -> None:
        """Test filter config serializer defaults."""
        filter_config = Filter()

        self.assertEqual(filter_config.serialize(), {})


class TestTransformConfig(unittest.TestCase):
    """Transform config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize transform config test case."""
        super().__init__(*args, **kwargs)

    def test_transform_constructor(self) -> None:
        """Test transform config constructor."""
        name = "TestTransform"
        data = {"shape": [1000, 224, 224, 3], "some_op": True}
        transform = Transform(name, data)

        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [1000, 224, 224, 3], "some_op": True},
        )

    def test_transform_constructor_without_params(self) -> None:
        """Test transform config constructor without parameters."""
        transform = Transform("TestTransform")

        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(transform.parameters, {})

    def test_transform_serializer(self) -> None:
        """Test transform config serializer."""
        name = "TestTransform"
        data = {"shape": [1000, 224, 224, 3], "some_op": True}
        transform = Transform(name, data)

        self.assertDictEqual(
            transform.serialize(),
            {"shape": [1000, 224, 224, 3], "some_op": True},
        )

    def test_transform_serializer_defaults(self) -> None:
        """Test transform config serializer defaults."""
        transform = Transform("TestTransform")

        self.assertDictEqual(transform.serialize(), {})


class TestDataloaderConfig(unittest.TestCase):
    """Dataloader config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize dataloader config test case."""
        super().__init__(*args, **kwargs)

    def test_dataloader_constructor(self) -> None:
        """Test dataloader config constructor."""
        data = {
            "last_batch": "rollover",
            "batch_size": 2,
            "dataset": {
                "TestDataset": {
                    "dataset_param": "/some/path",
                    "bool_param": True,
                    "list_param": ["item1", "item2"],
                },
            },
            "transform": {
                "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                "AnotherTestTransform": {"shape": [10, 299, 299, 3], "some_op": False},
            },
            "filter": {
                "LabelBalance": {"size": 1},
            },
        }
        dataloader = Dataloader(data)

        self.assertEqual(dataloader.last_batch, "rollover")
        self.assertEqual(dataloader.batch_size, 2)
        self.assertIsNotNone(dataloader.dataset)
        self.assertEqual(dataloader.dataset.name, "TestDataset")
        self.assertDictEqual(
            dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(dataloader.transform.items())[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(dataloader.transform.items())[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(dataloader.filter)
        self.assertIsNotNone(dataloader.filter.LabelBalance)
        self.assertEqual(dataloader.filter.LabelBalance.size, 1)

    def test_dataloader_constructor_fails_for_multiple_datasets(self) -> None:
        """Test dataloader config constructor."""
        data = {
            "last_batch": "rollover",
            "batch_size": 2,
            "dataset": {
                "TestDataset": {
                    "dataset_param": "/some/path",
                    "bool_param": True,
                    "list_param": ["item1", "item2"],
                },
                "TestDataset2": {
                    "dataset_param": "/some/path",
                    "bool_param": True,
                    "list_param": ["item1", "item2"],
                },
            },
            "transform": {
                "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                "AnotherTestTransform": {"shape": [10, 299, 299, 3], "some_op": False},
            },
            "filter": {
                "LabelBalance": {"size": 1},
            },
        }

        with self.assertRaisesRegex(
            ClientErrorException,
            "There can be specified only one dataset per dataloader. "
            "Found keys: TestDataset, TestDataset2.",
        ):
            Dataloader(data)

    def test_dataloader_constructor_defaults(self) -> None:
        """Test dataloader config constructor defaults."""
        dataloader = Dataloader()

        self.assertIsNone(dataloader.last_batch)
        self.assertEqual(dataloader.batch_size, 1)
        self.assertIsNone(dataloader.dataset)
        self.assertDictEqual(dataloader.transform, {})
        self.assertIsNone(dataloader.filter)

    def test_dataloader_constructor_batch_overwrite(self) -> None:
        """Test dataloader config constructor with batch overwrite."""
        dataloader = Dataloader(
            data={"batch_size": 2},
            batch_size=32,
        )
        self.assertIsNone(dataloader.last_batch)
        self.assertEqual(
            dataloader.batch_size,
            32,
        )  # Batch size from parameter has higher priority
        self.assertIsNone(dataloader.dataset)
        self.assertDictEqual(dataloader.transform, {})
        self.assertIsNone(dataloader.filter)

    def test_dataloader_constructor_defaults_batch_overwrite(self) -> None:
        """Test dataloader config constructor defaults with batch overwrite."""
        dataloader = Dataloader(batch_size=8)

        self.assertIsNone(dataloader.last_batch)
        self.assertEqual(dataloader.batch_size, 8)
        self.assertIsNone(dataloader.dataset)
        self.assertDictEqual(dataloader.transform, {})
        self.assertIsNone(dataloader.filter)

    def test_dataloader_serializer(self) -> None:
        """Test dataloader config serializer."""
        data = {
            "last_batch": "rollover",
            "batch_size": 2,
            "dataset": {
                "TestDataset": {
                    "dataset_param": "/some/path",
                    "bool_param": True,
                    "list_param": ["item1", "item2"],
                },
            },
            "transform": {
                "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                "AnotherTestTransform": {"shape": [10, 299, 299, 3], "some_op": False},
            },
            "filter": {
                "LabelBalance": {"size": 1},
            },
        }
        dataloader = Dataloader(data)

        self.assertDictEqual(
            dataloader.serialize(),
            {
                "last_batch": "rollover",
                "batch_size": 2,
                "dataset": {
                    "TestDataset": {
                        "dataset_param": "/some/path",
                        "bool_param": True,
                        "list_param": ["item1", "item2"],
                    },
                },
                "transform": {
                    "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                    "AnotherTestTransform": {
                        "shape": [10, 299, 299, 3],
                        "some_op": False,
                    },
                },
                "filter": {
                    "LabelBalance": {"size": 1},
                },
            },
        )

    def test_dataloader_serializer_defaults(self) -> None:
        """Test dataloader config serializer defaults."""
        dataloader = Dataloader()
        self.assertDictEqual(
            dataloader.serialize(),
            {
                "batch_size": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()

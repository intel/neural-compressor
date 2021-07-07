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
"""Quantization config test."""

import unittest

from lpot.ux.utils.workload.quantization import (
    Advance,
    Calibration,
    Quantization,
    WiseConfig,
    WiseConfigDetails,
)


class TestCalibrationConfig(unittest.TestCase):
    """Calibration config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Calibration config test constructor."""
        super().__init__(*args, **kwargs)

    def test_calibration_constructor(self) -> None:
        """Test Calibration config constructor."""
        data = {
            "sampling_size": "10, 50, 100, 200",
            "dataloader": {
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
        }
        calibration = Calibration(data)

        self.assertEqual(calibration.sampling_size, "10, 50, 100, 200")
        self.assertEqual(calibration.dataloader.last_batch, "rollover")
        self.assertEqual(
            calibration.dataloader.batch_size,
            1,
        )  # Calibration batch size should be always set to 1
        self.assertIsNotNone(calibration.dataloader.dataset)
        self.assertEqual(calibration.dataloader.dataset.name, "TestDataset")
        self.assertDictEqual(
            calibration.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(calibration.dataloader.transform.items())[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(calibration.dataloader.transform.items())[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(calibration.dataloader.filter)
        self.assertIsNotNone(calibration.dataloader.filter.LabelBalance)
        self.assertEqual(calibration.dataloader.filter.LabelBalance.size, 1)

    def test_calibration_constructor_defaults(self) -> None:
        """Test Calibration config constructor defaults."""
        calibration = Calibration()

        self.assertEqual(calibration.sampling_size, 100)
        self.assertIsNone(calibration.dataloader)

    def test_calibration_serializer(self) -> None:
        """Test Calibration config serializer."""
        data = {
            "sampling_size": "10, 50, 100, 200",
            "dataloader": {
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
                "some_additional_field": {
                    "not_important_data": True,
                },
            },
        }
        calibration = Calibration(data)
        result = calibration.serialize()

        self.assertDictEqual(
            result,
            {
                "sampling_size": "10, 50, 100, 200",
                "dataloader": {
                    "last_batch": "rollover",
                    "batch_size": 1,
                    "dataset": {
                        "TestDataset": {
                            "dataset_param": "/some/path",
                            "bool_param": True,
                            "list_param": ["item1", "item2"],
                        },
                    },
                    "transform": {
                        "TestTransform": {
                            "shape": [1000, 224, 224, 3],
                            "some_op": True,
                        },
                        "AnotherTestTransform": {
                            "shape": [10, 299, 299, 3],
                            "some_op": False,
                        },
                    },
                    "filter": {
                        "LabelBalance": {"size": 1},
                    },
                },
            },
        )


class TestWiseConfigDetailsConfig(unittest.TestCase):
    """WiseConfigDetails config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize WiseConfigDetails config test."""
        super().__init__(*args, **kwargs)

    def test_wise_config_details_constructor(self) -> None:
        """Test WiseConfigDetails config constructor."""
        data = {
            "granularity": "per_channel",
            "scheme": "asym",
            "dtype": "bf16",
            "algorithm": "kl",
        }
        wise_config_details = WiseConfigDetails(data)

        self.assertEqual(wise_config_details.granularity, "per_channel")
        self.assertEqual(wise_config_details.scheme, "asym")
        self.assertEqual(wise_config_details.dtype, "bf16")
        self.assertEqual(wise_config_details.algorithm, "kl")

    def test_wise_config_details_constructor_defaults(self) -> None:
        """Test WiseConfigDetails config constructor defaults."""
        wise_config_details = WiseConfigDetails()

        self.assertIsNone(wise_config_details.granularity)
        self.assertIsNone(wise_config_details.scheme)
        self.assertIsNone(wise_config_details.dtype)
        self.assertIsNone(wise_config_details.algorithm)

    def test_wise_config_details_serializer(self) -> None:
        """Test WiseConfigDetails config serializer."""
        data = {
            "granularity": "per_channel",
            "scheme": "asym",
            "dtype": "bf16",
            "algorithm": "kl",
            "output_dtype": "int8",
            "wise": True,
        }
        wise_config_details = WiseConfigDetails(data)
        result = wise_config_details.serialize()

        self.assertDictEqual(
            result,
            {
                "granularity": "per_channel",
                "scheme": "asym",
                "dtype": "bf16",
                "algorithm": "kl",
            },
        )


class TestWiseConfigConfig(unittest.TestCase):
    """WiseConfig config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize WiseConfig config test."""
        super().__init__(*args, **kwargs)

    def test_wise_config_constructor(self) -> None:
        """Test WiseConfig config constructor."""
        data = {
            "weight": {
                "granularity": "per_channel",
                "scheme": "asym",
                "dtype": "bf16",
                "algorithm": "minmax",
            },
            "activation": {
                "granularity": "per_tensor",
                "scheme": "sym",
                "dtype": "int8",
                "algorithm": "minmax",
            },
        }
        wise_config = WiseConfig(data)

        self.assertIsNotNone(wise_config.weight)
        self.assertEqual(wise_config.weight.granularity, "per_channel")
        self.assertEqual(wise_config.weight.scheme, "asym")
        self.assertEqual(wise_config.weight.dtype, "bf16")
        self.assertEqual(wise_config.weight.algorithm, "minmax")

        self.assertIsNotNone(wise_config.activation)
        self.assertEqual(wise_config.activation.granularity, "per_tensor")
        self.assertEqual(wise_config.activation.scheme, "sym")
        self.assertEqual(wise_config.activation.dtype, "int8")
        self.assertEqual(wise_config.activation.algorithm, "minmax")

    def test_wise_config_constructor_defaults(self) -> None:
        """Test WiseConfig config constructor defaults."""
        wise_config = WiseConfig()

        self.assertIsNone(wise_config.weight)
        self.assertIsNone(wise_config.activation)

    def test_wise_config_serializer(self) -> None:
        """Test WiseConfig config serializer."""
        data = {
            "weight": {
                "granularity": "per_channel",
                "scheme": "asym",
                "dtype": "bf16",
                "algorithm": "minmax",
            },
            "activation": {
                "granularity": "per_tensor",
                "scheme": "sym",
                "dtype": "int8",
                "algorithm": "minmax",
            },
            "additional_field": {"asd": 123},
        }
        wise_config = WiseConfig(data)
        result = wise_config.serialize()

        self.assertDictEqual(
            result,
            {
                "weight": {
                    "granularity": "per_channel",
                    "scheme": "asym",
                    "dtype": "bf16",
                    "algorithm": "minmax",
                },
                "activation": {
                    "granularity": "per_tensor",
                    "scheme": "sym",
                    "dtype": "int8",
                    "algorithm": "minmax",
                },
            },
        )


class TestAdvanceConfig(unittest.TestCase):
    """Advance config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Advance config test constructor."""
        super().__init__(*args, **kwargs)

    def test_advance_constructor(self) -> None:
        """Test Advance config constructor."""
        data = {"bias_correction": "weight_empirical"}
        advance = Advance(data)

        self.assertEqual(advance.bias_correction, "weight_empirical")

    def test_advance_constructor_defaults(self) -> None:
        """Test Advance config constructor defaults."""
        advance = Advance()

        self.assertIsNone(advance.bias_correction)


class TestQuantizationConfig(unittest.TestCase):
    """Quantization config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Quantization config test constructor."""
        super().__init__(*args, **kwargs)

    def test_quantization_constructor(self) -> None:
        """Test Quantization config constructor."""
        data = {
            "calibration": {
                "sampling_size": "10, 50, 100, 200",
                "dataloader": {
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
                        "TestTransform": {
                            "shape": [1000, 224, 224, 3],
                            "some_op": True,
                        },
                        "AnotherTestTransform": {
                            "shape": [10, 299, 299, 3],
                            "some_op": False,
                        },
                    },
                    "filter": {
                        "LabelBalance": {"size": 1},
                    },
                },
            },
            "model_wise": {
                "weight": {
                    "granularity": "per_channel_model",
                    "scheme": "asym_model",
                    "dtype": "bf16_model",
                    "algorithm": "kl_model",
                },
                "activation": {
                    "granularity": "per_tensor_model",
                    "scheme": "sym_model",
                    "dtype": "int8_model",
                    "algorithm": "minmax_model",
                },
            },
            "op_wise": {
                "weight": {
                    "granularity": "per_channel_op",
                    "scheme": "asym_op",
                    "dtype": "bf16_op",
                    "algorithm": "kl_op",
                },
                "activation": {
                    "granularity": "per_tensor_op",
                    "scheme": "sym_op",
                    "dtype": "int8_op",
                    "algorithm": "minmax_op",
                },
            },
            "approach": "quant_aware_training",
            "advance": {"bias_correction": "weight_empirical"},
        }
        quantization = Quantization(data)

        self.assertIsNotNone(quantization.calibration)
        self.assertEqual(quantization.calibration.sampling_size, "10, 50, 100, 200")
        self.assertEqual(quantization.calibration.dataloader.last_batch, "rollover")
        self.assertEqual(
            quantization.calibration.dataloader.batch_size,
            1,
        )  # Calibration batch size should be always set to 1
        self.assertIsNotNone(quantization.calibration.dataloader.dataset)
        self.assertEqual(
            quantization.calibration.dataloader.dataset.name,
            "TestDataset",
        )
        self.assertDictEqual(
            quantization.calibration.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(
            quantization.calibration.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(
            quantization.calibration.dataloader.transform.items(),
        )[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(quantization.calibration.dataloader.filter)
        self.assertIsNotNone(
            quantization.calibration.dataloader.filter.LabelBalance,
        )
        self.assertEqual(
            quantization.calibration.dataloader.filter.LabelBalance.size,
            1,
        )

        self.assertIsNotNone(quantization.model_wise)
        self.assertIsNotNone(quantization.model_wise.weight)
        self.assertEqual(
            quantization.model_wise.weight.granularity,
            "per_channel_model",
        )
        self.assertEqual(quantization.model_wise.weight.scheme, "asym_model")
        self.assertEqual(quantization.model_wise.weight.dtype, "bf16_model")
        self.assertEqual(quantization.model_wise.weight.algorithm, "kl_model")
        self.assertIsNotNone(quantization.model_wise.activation)
        self.assertEqual(
            quantization.model_wise.activation.granularity,
            "per_tensor_model",
        )
        self.assertEqual(quantization.model_wise.activation.scheme, "sym_model")
        self.assertEqual(quantization.model_wise.activation.dtype, "int8_model")
        self.assertEqual(quantization.model_wise.activation.algorithm, "minmax_model")

        self.assertIsNotNone(quantization.op_wise)
        self.assertDictEqual(
            quantization.op_wise,
            {
                "weight": {
                    "granularity": "per_channel_op",
                    "scheme": "asym_op",
                    "dtype": "bf16_op",
                    "algorithm": "kl_op",
                },
                "activation": {
                    "granularity": "per_tensor_op",
                    "scheme": "sym_op",
                    "dtype": "int8_op",
                    "algorithm": "minmax_op",
                },
            },
        )

        self.assertEqual(quantization.approach, "quant_aware_training")

        self.assertIsNotNone(quantization.advance)
        self.assertEqual(quantization.advance.bias_correction, "weight_empirical")

    def test_quantization_constructor_defaults(self) -> None:
        """Test Quantization config constructor defaults."""
        quantization = Quantization()

        self.assertIsNone(quantization.calibration)
        self.assertIsNone(quantization.model_wise)
        self.assertIsNone(quantization.op_wise)
        self.assertEqual(quantization.approach, "post_training_static_quant")
        self.assertIsNone(quantization.advance)


if __name__ == "__main__":
    unittest.main()

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
"""Database operations test."""

import unittest
from unittest.mock import patch

from neural_compressor.ux.components.db_manager.db_operations import (
    DatasetAPIInterface,
    DiagnosisAPIInterface,
)
from neural_compressor.ux.utils.consts import DomainFlavours, Domains, Frameworks
from neural_compressor.ux.utils.exceptions import ClientErrorException


@patch("sys.argv", ["inc_bench.py", "-p5000"])
class TestDatasetAPIInterface(unittest.TestCase):
    """Test class for Dataset API Interface."""

    def test_get_predefined_dataset_without_params(self) -> None:
        """Test getting predefined dataset when params are not set fails."""
        data: dict = {}
        with self.assertRaisesRegex(
            ClientErrorException,
            "Could not find required parameter. Required keys are .*",
        ):
            DatasetAPIInterface.get_predefined_dataset(data)

    def test_get_predefined_tensorflow_image_recognition_dataset(self) -> None:
        """Test getting predefined dataset for TensorFlow Image Recognition model."""
        data: dict = {
            "framework": Frameworks.TF.value,
            "domain": Domains.IMAGE_RECOGNITION.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": [
                {
                    "name": "ResizeCropImagenet",
                    "params": [
                        {
                            "name": "height",
                            "value": 224,
                        },
                        {
                            "name": "width",
                            "value": 224,
                        },
                        {
                            "name": "mean_value",
                            "value": [
                                123.68,
                                116.78,
                                103.94,
                            ],
                        },
                    ],
                },
            ],
            "dataloader": {
                "name": "ImageRecord",
                "params": [
                    {
                        "name": "root",
                        "value": "/path/to/calibration/dataset",
                    },
                ],
            },
            "metric": "topk",
            "metric_param": 1,
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_tensorflow_object_detection_dataset(self) -> None:
        """Test getting predefined dataset for TensorFlow Object Detection model."""
        data: dict = {
            "framework": Frameworks.TF.value,
            "domain": Domains.OBJECT_DETECTION.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": [],
            "dataloader": {
                "name": "COCORecord",
                "params": [
                    {
                        "name": "root",
                        "value": "/path/to/calibration/dataset",
                    },
                ],
            },
            "metric": "COCOmAP",
            "metric_param": {},
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_tensorflow_object_detection_ssd_dataset(self) -> None:
        """Test getting predefined dataset for TensorFlow Object Detection SSD model."""
        data: dict = {
            "framework": Frameworks.TF.value,
            "domain": Domains.OBJECT_DETECTION.value,
            "domain_flavour": DomainFlavours.SSD.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": [
                {
                    "name": "Rescale",
                    "params": [],
                },
                {
                    "name": "Normalize",
                    "params": [
                        {
                            "name": "mean",
                            "value": [
                                0.485,
                                0.456,
                                0.406,
                            ],
                        },
                        {
                            "name": "std",
                            "value": [
                                0.229,
                                0.224,
                                0.225,
                            ],
                        },
                    ],
                },
                {
                    "name": "Resize",
                    "params": [
                        {
                            "name": "size",
                            "value": [
                                1200,
                                1200,
                            ],
                        },
                    ],
                },
            ],
            "dataloader": {
                "name": "COCORecord",
                "params": [
                    {
                        "name": "root",
                        "value": "/path/to/calibration/dataset",
                    },
                ],
            },
            "metric": "COCOmAP",
            "metric_param": {},
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_tensorflow_nlp_dataset(self) -> None:
        """Test getting predefined dataset for TensorFlow NLP model."""
        data: dict = {
            "framework": Frameworks.TF.value,
            "domain": Domains.NLP.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": [],
            "dataloader": {
                "name": "bert",
                "params": [
                    {
                        "name": "root",
                        "value": "/path/to/eval.tf_record",
                    },
                    {
                        "name": "label_file",
                        "value": "/path/to/dev-v1.1.json",
                    },
                ],
            },
            "metric": "SquadF1",
            "metric_param": {},
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_tensorflow_recommendation_dataset(self) -> None:
        """Test getting predefined dataset for TensorFlow Recommendation model."""
        data: dict = {
            "framework": Frameworks.TF.value,
            "domain": Domains.RECOMMENDATION.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": {},
            "dataloader": {},
            "metric": {},
            "metric_param": {},
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_onnx_image_recognition_dataset(self) -> None:
        """Test getting predefined dataset for ONNX Image Recognition model."""
        data: dict = {
            "framework": Frameworks.ONNX.value,
            "domain": Domains.IMAGE_RECOGNITION.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": [
                {
                    "name": "ResizeCropImagenet",
                    "params": [
                        {
                            "name": "height",
                            "value": 224,
                        },
                        {
                            "name": "width",
                            "value": 224,
                        },
                        {
                            "name": "mean_value",
                            "value": [
                                0.485,
                                0.456,
                                0.406,
                            ],
                        },
                    ],
                },
            ],
            "dataloader": {
                "name": "ImagenetRaw",
                "params": [
                    {
                        "name": "data_path",
                        "value": "/path/to/calibration/dataset",
                    },
                    {
                        "name": "image_list",
                        "value": "/path/to/calibration/label",
                    },
                ],
            },
            "metric": "topk",
            "metric_param": 1,
        }
        self.assertDictEqual(actual, expected)

    def test_get_predefined_onnx_nlp_dataset(self) -> None:
        """Test getting predefined dataset for ONNX NLP model."""
        data: dict = {
            "framework": Frameworks.ONNX.value,
            "domain": Domains.NLP.value,
            "domain_flavour": DomainFlavours.NONE.value,
        }
        actual = DatasetAPIInterface.get_predefined_dataset(data)

        expected: dict = {
            "transform": {},
            "dataloader": {},
            "metric": {},
            "metric_param": {},
        }
        self.assertDictEqual(actual, expected)


@patch("sys.argv", ["inc_bench.py", "-p5000"])
class TestDiagnosisAPIInterface(unittest.TestCase):
    """Test class for Dataset API Interface."""

    def test_parse_op_wise_config(self) -> None:
        """Test parsing op wise config."""
        data: dict = {
            "MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D": {
                "pattern": {
                    "precision": "float32",
                },
            },
            "MobilenetV1/MobilenetV1/Conv2d_0/Conv2D": {
                "pattern": {
                    "precision": "bfloat16",
                },
                "weight": {
                    "granularity": "per_tensor",
                },
            },
        }

        parsed_data = DiagnosisAPIInterface.parse_op_wise_config(data)
        expected = {
            "MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D": {
                "weight": {
                    "dtype": ["float32"],
                },
                "activation": {
                    "dtype": ["float32"],
                },
            },
            "MobilenetV1/MobilenetV1/Conv2d_0/Conv2D": {
                "weight": {
                    "dtype": ["bfloat16"],
                    "granularity": ["per_tensor"],
                },
                "activation": {
                    "dtype": ["bfloat16"],
                },
            },
        }

        self.assertDictEqual(parsed_data, expected)

    def test_parse_model_wise_config(self) -> None:
        """Test parsing model wise config."""
        data: dict = {"weight": {"scheme": "sym", "bit": 7}, "activation": {"algorithm": "minmax"}}

        parsed_data = DiagnosisAPIInterface.parse_model_wise_config(data)

        expected = {
            "weight": {
                "scheme": ["sym"],
                "bit": [7],
            },
            "activation": {
                "algorithm": ["minmax"],
            },
        }

        self.assertDictEqual(parsed_data, expected)
